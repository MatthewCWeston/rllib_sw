import numpy as np

import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F

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

    @torch.no_grad()
    def select_action_batch(self, obs_batch):
        """Batched inference for N envs at once."""
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
        
        # Normalize advantages 
        t_advantages -= t_advantages.mean()
        t_advantages /= t_advantages.std()

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