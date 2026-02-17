# sw_main.py
import numpy as np
import torch
import os
import multiprocessing as mp

from sw_curriculum import curriculum_stages
from sw_agent import SpaceWarNet
from sw_logging import TrainingDashboard
from sw_env import SW_1v1_env_singleplayer, BASE_REWARD
from sw_ppo import RolloutBuffer, PPOTrainer


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

def train(num_workers=4, batch_size=32768):
    BATCH_SIZE = batch_size
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
    parser.add_argument("--batch-size", type=int, default=32768,
                        help="Steps to sample before updating")
    args = parser.parse_args()

    train(num_workers=args.num_workers, batch_size=args.batch_size)