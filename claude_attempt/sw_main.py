# sw_main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os
import multiprocessing as mp

from sw_curriculum import curriculum_stages
from sw_agent import SpaceWarNet
from sw_logging import TrainingDashboard
from sw_env import SW_1v1_env_singleplayer, BASE_REWARD


class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, obs, action, log_prob, value, reward, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.rewards)

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()


class PPOTrainer:
    def __init__(self, model, lr=3e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, epochs=4, minibatch_size=256,
                 entropy_coef=0.01, vf_coef=0.5, max_grad_norm=0.5):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0.0
        values_ext = values + [0.0]
        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values_ext[t + 1] * non_terminal - values_ext[t]
            gae = delta + self.gamma * self.lam * non_terminal * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, values)]
        return advantages, returns

    def select_action_batch(self, obs_batch):
        """Batched inference for N envs at once."""
        with torch.no_grad():
            logits_list, values = self.model(obs_batch)

        batch_size = values.shape[0]
        all_actions = []
        total_log_probs = torch.zeros(batch_size)

        for logits in logits_list:
            dist = Categorical(logits=logits)
            actions = dist.sample()
            total_log_probs += dist.log_prob(actions)
            all_actions.append(actions)

        # all_actions: list of [N] tensors, one per action head
        # Stack to [N, num_heads]
        actions_stacked = torch.stack(all_actions, dim=1)

        return (
            actions_stacked.tolist(),        # list of lists
            total_log_probs.tolist(),        # list of floats
            values.squeeze(-1).tolist(),     # list of floats
        )

    @staticmethod
    def _collate_obs(obs_list):
        batched = {}
        for key in obs_list[0]:
            batched[key] = torch.stack([o[key] for o in obs_list], dim=0)
        return batched

    def update(self, buffers):
        """Run PPO update from a list of per-env buffers (GAE per env)."""
        all_obs = []
        all_actions = []
        all_log_probs = []
        all_advantages = []
        all_returns = []

        for b in buffers:
            if len(b) == 0:
                continue
            adv, ret = self.compute_gae(b.rewards, b.values, b.dones)
            all_obs.extend(b.obs)
            all_actions.extend(b.actions)
            all_log_probs.extend(b.log_probs)
            all_advantages.extend(adv)
            all_returns.extend(ret)

        n = len(all_obs)
        if n == 0:
            return None

        obs_batched = self._collate_obs(all_obs)
        t_actions = torch.tensor(all_actions, dtype=torch.long)
        t_log_probs = torch.tensor(all_log_probs, dtype=torch.float32)
        t_advantages = torch.tensor(all_advantages, dtype=torch.float32)
        t_returns = torch.tensor(all_returns, dtype=torch.float32)

        indices = np.arange(n)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n_updates": 0}

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.minibatch_size):
                end = min(start + self.minibatch_size, n)
                mb_idx = torch.tensor(indices[start:end], dtype=torch.long)

                obs_batch = {k: v[mb_idx] for k, v in obs_batched.items()}
                actions = t_actions[mb_idx]
                old_lp = t_log_probs[mb_idx]
                adv = t_advantages[mb_idx]
                ret = t_returns[mb_idx]

                logits_list, values = self.model(obs_batch)
                new_lp = torch.zeros(actions.shape[0])
                entropy = torch.zeros(1)
                for i, logits in enumerate(logits_list):
                    dist = Categorical(logits=logits)
                    new_lp = new_lp + dist.log_prob(actions[:, i])
                    entropy = entropy + dist.entropy().mean()

                adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = (new_lp - old_lp).exp()
                surr1 = ratio * adv_norm
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_norm
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), ret)
                loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["n_updates"] += 1

        for b in buffers:
            b.clear()

        return stats


def tensorize_obs(obs):
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}


def make_env(stage):
    return SW_1v1_env_singleplayer({
        "egocentric": True,
        "speed": 5.0,
        "ep_length": 4096,
        **curriculum_stages[stage],
    })


# ─── Persistent env worker ───────────────────────────────────────────────────

def env_worker(pipe, initial_stage):
    """
    Owns one env. Only does env.step() and env.reset().
    Protocol:
        recv ("step", action)  -> send (obs_dict, reward, done, base_reward)
        recv ("reset", stage)  -> send (obs_dict,)
        recv ("close",)        -> exit
    """
    stage = initial_stage
    env = make_env(stage)
    obs, _ = env.reset()
    # Send initial obs
    pipe.send((obs[0],))

    while True:
        msg = pipe.recv()
        cmd = msg[0]

        if cmd == "step":
            action = msg[1]
            next_obs, rewards, term, trunc, info = env.step({0: action})
            reward = rewards[0]
            done = term["__all__"] or trunc["__all__"]
            base_reward = info[BASE_REWARD]

            if done:
                env = make_env(stage)
                obs, _ = env.reset()
                pipe.send((obs[0], reward, True, base_reward))
            else:
                pipe.send((next_obs[0], reward, False, base_reward))

        elif cmd == "reset":
            stage = msg[1]
            env = make_env(stage)
            obs, _ = env.reset()
            pipe.send((obs[0],))

        elif cmd == "close":
            pipe.close()
            return


class ParallelEnvManager:
    def __init__(self, num_envs, stage):
        self.num_envs = num_envs
        self.pipes = []
        self.processes = []
        self.current_obs = []

        for _ in range(num_envs):
            parent_pipe, child_pipe = mp.Pipe()
            p = mp.Process(target=env_worker, args=(child_pipe, stage), daemon=True)
            p.start()
            child_pipe.close()
            self.pipes.append(parent_pipe)
            self.processes.append(p)

        for i in range(num_envs):
            (obs_dict,) = self.pipes[i].recv()
            self.current_obs.append(tensorize_obs(obs_dict))

    def step_async(self, actions_per_env):
        """Send actions to all workers (non-blocking)."""
        for i in range(self.num_envs):
            self.pipes[i].send(("step", actions_per_env[i]))

    def step_wait(self):
        """Collect results from all workers."""
        results = []
        for i in range(self.num_envs):
            resp = self.pipes[i].recv()
            obs_dict, reward, done, base_reward = resp
            obs_tensor = tensorize_obs(obs_dict)
            self.current_obs[i] = obs_tensor
            results.append((obs_tensor, reward, done, base_reward))
        return results

    def get_batched_obs(self):
        """Stack current obs across all envs into a single batched dict."""
        batched = {}
        for key in self.current_obs[0]:
            batched[key] = torch.stack([o[key] for o in self.current_obs], dim=0)
        return batched

    def set_stage(self, stage):
        for i in range(self.num_envs):
            self.pipes[i].send(("reset", stage))
        for i in range(self.num_envs):
            (obs_dict,) = self.pipes[i].recv()
            self.current_obs[i] = tensorize_obs(obs_dict)

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("close",))
            except BrokenPipeError:
                pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


# ─── Main training loop ──────────────────────────────────────────────────────

def train(num_workers=4):
    BATCH_SIZE = 32768
    SAVE_INTERVAL = 50
    ADVANCEMENT_THRESHOLD = 0.4
    ADVANCEMENT_WINDOW = 100
    TOTAL_UPDATES = 10_000

    os.makedirs("checkpoints", exist_ok=True)

    model = SpaceWarNet()
    trainer = PPOTrainer(
        model, lr=3e-4, gamma=0.99, lam=0.95,
        clip_eps=0.2, epochs=4, minibatch_size=256,
        entropy_coef=0.01, vf_coef=0.5,
    )

    stage = 0
    dash = TrainingDashboard(
        total_updates=TOTAL_UPDATES,
        advancement_window=ADVANCEMENT_WINDOW,
        log_file="training_log.csv",
    )
    dash.set_stage(stage, curriculum_stages[stage])
    dash.log_event(f"Starting training - stage 0 with {num_workers} workers")

    env_mgr = ParallelEnvManager(num_workers, stage)
    update_count = 0

    buffers = [RolloutBuffer() for _ in range(num_workers)]
    ep_base_rewards = [0.0] * num_workers
    ep_shaped_rewards = [0.0] * num_workers

    while update_count < TOTAL_UPDATES:
        steps_collected = 0

        while steps_collected < BATCH_SIZE:
            # ── Batched inference: one forward pass for all envs ──────
            obs_batch = env_mgr.get_batched_obs()
            actions_list, log_probs_list, values_list = trainer.select_action_batch(obs_batch)

            # Save pre-step obs references
            pre_step_obs = [env_mgr.current_obs[i] for i in range(num_workers)]

            # ── Send all actions at once, then wait ───────────────────
            env_mgr.step_async(actions_list)
            results = env_mgr.step_wait()

            # ── Store transitions per env ─────────────────────────────
            for i, (obs_tensor, reward, done, base_reward) in enumerate(results):
                buffers[i].add(
                    pre_step_obs[i],
                    actions_list[i],
                    log_probs_list[i],
                    values_list[i],
                    reward,
                    done,
                )
                ep_base_rewards[i] += base_reward
                ep_shaped_rewards[i] += reward

                if done:
                    dash.log_episode(ep_base_rewards[i], ep_shaped_rewards[i])
                    ep_base_rewards[i] = 0.0
                    ep_shaped_rewards[i] = 0.0

            steps_collected += num_workers

        # ── Check curriculum advancement ──────────────────────────────
        if len(dash.base_reward_history) >= ADVANCEMENT_WINDOW:
            mean_base = np.mean(dash.base_reward_history)
            if mean_base > ADVANCEMENT_THRESHOLD and stage < len(curriculum_stages) - 1:
                stage += 1
                dash.base_reward_history.clear()
                dash.shaped_reward_history.clear()
                dash.log_stage_advance(stage, curriculum_stages[stage], mean_base)
                env_mgr.set_stage(stage)
                for b in buffers:
                    b.clear()
                ep_base_rewards = [0.0] * num_workers
                ep_shaped_rewards = [0.0] * num_workers
                continue

        # ── PPO update (GAE computed per-env, then merged) ────────────
        stats = trainer.update(buffers)
        if stats is None:
            continue

        update_count += 1
        dash.log_update(stats)

        if update_count % SAVE_INTERVAL == 0:
            path = f"checkpoints/spacewar_stage{stage}_update{update_count}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "stage": stage,
                "update_count": update_count,
                "ep_count": dash.ep_count,
            }, path)
            dash.log_save(path)

    env_mgr.close()
    dash.log_event("Training complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(),
                        help="Number of parallel env workers (default: all CPUs)")
    args = parser.parse_args()

    train(num_workers=args.num_workers)