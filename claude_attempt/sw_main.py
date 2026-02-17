# sw_main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import numpy as np
import os

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

    def select_action(self, obs_dict):
        with torch.no_grad():
            logits_list, value = self.model(obs_dict)
        actions = []
        log_probs = []
        for logits in logits_list:
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample()
            actions.append(action.item())
            log_probs.append(dist.log_prob(action))
        total_log_prob = sum(log_probs)
        return actions, total_log_prob.item(), value.squeeze().item()

    @staticmethod
    def _collate_obs(obs_list):
        batched = {}
        for key in obs_list[0]:
            batched[key] = torch.stack([o[key] for o in obs_list], dim=0)
        return batched

    def update(self, buffer):
        advantages, returns = self.compute_gae(
            buffer.rewards, buffer.values, buffer.dones
        )
        all_obs = self._collate_obs(buffer.obs)
        all_actions = torch.tensor(buffer.actions, dtype=torch.long)
        all_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32)
        all_advantages = torch.tensor(advantages, dtype=torch.float32)
        all_returns = torch.tensor(returns, dtype=torch.float32)

        n = len(buffer)
        indices = np.arange(n)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n_updates": 0}

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.minibatch_size):
                end = min(start + self.minibatch_size, n)
                mb_idx = torch.tensor(indices[start:end], dtype=torch.long)

                obs_batch = {k: v[mb_idx] for k, v in all_obs.items()}
                actions = all_actions[mb_idx]
                old_lp = all_log_probs[mb_idx]
                adv = all_advantages[mb_idx]
                ret = all_returns[mb_idx]

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

        buffer.clear()
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


def train():
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
    buffer = RolloutBuffer()
    ep_base_reward = 0.0
    ep_shaped_reward = 0.0

    dash = TrainingDashboard(
        total_updates=TOTAL_UPDATES,
        advancement_window=ADVANCEMENT_WINDOW,
        log_file="training_log.csv",
    )
    dash.set_stage(stage, curriculum_stages[stage])
    dash.log_event(f"Starting training - stage 0")

    env = make_env(stage)
    obs, _ = env.reset()
    obs_tensor = tensorize_obs(obs[0])
    update_count = 0

    while update_count < TOTAL_UPDATES:
        for _ in range(BATCH_SIZE):
            action, log_prob, value = trainer.select_action(obs_tensor)
            next_obs, rewards, term, trunc, info = env.step({0: action})

            reward = rewards[0]
            done = term["__all__"] or trunc["__all__"]

            buffer.add(obs_tensor, action, log_prob, value, reward, done)
            ep_base_reward += info[BASE_REWARD]
            ep_shaped_reward += reward

            if done:
                dash.log_episode(ep_base_reward, ep_shaped_reward)
                ep_base_reward = 0.0
                ep_shaped_reward = 0.0

                if len(dash.base_reward_history) >= ADVANCEMENT_WINDOW:
                    mean_base = np.mean(dash.base_reward_history)
                    if mean_base > ADVANCEMENT_THRESHOLD and stage < len(curriculum_stages) - 1:
                        stage += 1
                        dash.base_reward_history.clear()
                        dash.shaped_reward_history.clear()
                        dash.log_stage_advance(stage, curriculum_stages[stage], mean_base)

                env = make_env(stage)
                obs, _ = env.reset()
                obs_tensor = tensorize_obs(obs[0])
            else:
                obs = next_obs
                obs_tensor = tensorize_obs(obs[0])

        if not dash.base_reward_history:
            continue

        stats = trainer.update(buffer)
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

    dash.log_event("Training complete!")


if __name__ == "__main__":
    train()