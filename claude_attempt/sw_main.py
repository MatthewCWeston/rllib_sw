# sw_main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import numpy as np
import copy

from sw_curriculum import curriculum_stages
from sw_agent import SpaceWarNet
from sw_env import SW_1v1_env_singleplayer, BASE_REWARD


class RolloutBuffer:
    """Stores transitions from multiple episodes until we have enough steps to update."""
    def __init__(self):
        self.obs = []          # list of obs dicts
        self.actions = []      # list of [a0, a1, a2]
        self.log_probs = []    # list of float
        self.values = []       # list of float
        self.rewards = []      # list of float
        self.dones = []        # list of bool

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
        """Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        # Append a bootstrap value of 0 (episode boundary)
        values_ext = values + [0.0]
        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values_ext[t + 1] * non_terminal - values_ext[t]
            gae = delta + self.gamma * self.lam * non_terminal * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, values)]
        return advantages, returns

    def select_action(self, obs_dict):
        """Sample action from policy and return action, log_prob, value."""
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

    def _collate_obs(self, obs_list):
        """Stack a list of obs dicts into batched tensors."""
        batched = {}
        for key in obs_list[0]:
            batched[key] = torch.stack([o[key] for o in obs_list], dim=0)
        return batched

    def make_batches(self, buffer):
        """
        Compute GAE, then yield shuffled minibatches for PPO update.
        
        Returns a generator of (obs_batch, actions, old_log_probs, advantages, returns)
        """
        advantages, returns = self.compute_gae(
            buffer.rewards, buffer.values, buffer.dones
        )

        # Tensorize everything
        all_obs = self._collate_obs(buffer.obs)
        all_actions = torch.tensor(buffer.actions, dtype=torch.long)         # (N, 3)
        all_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32)  # (N,)
        all_advantages = torch.tensor(advantages, dtype=torch.float32)       # (N,)
        all_returns = torch.tensor(returns, dtype=torch.float32)             # (N,)

        n = len(buffer)
        indices = np.arange(n)

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.minibatch_size):
                end = min(start + self.minibatch_size, n)
                mb_idx = indices[start:end]
                mb_idx_t = torch.tensor(mb_idx, dtype=torch.long)

                obs_batch = {k: v[mb_idx_t] for k, v in all_obs.items()}
                yield (
                    obs_batch,
                    all_actions[mb_idx_t],
                    all_log_probs[mb_idx_t],
                    all_advantages[mb_idx_t],
                    all_returns[mb_idx_t],
                )

    def update(self, buffer):
        """Run PPO update over the full rollout buffer."""
        stats = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "n_updates": 0}

        for batch in self.make_batches(buffer):
            obs, actions, old_log_probs, advantages, returns = batch

            logits_list, values = self.model(obs)

            # Compute new log probs and entropy for each action dimension
            new_log_prob = torch.zeros(actions.shape[0])
            entropy = torch.zeros(1)
            for i, logits in enumerate(logits_list):
                dist = Categorical(logits=logits)
                new_log_prob = new_log_prob + dist.log_prob(actions[:, i])
                entropy = entropy + dist.entropy().mean()

            # Normalize advantages within the minibatch
            adv_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO clipped objective
            ratio = (new_log_prob - old_log_probs).exp()
            surr1 = ratio * adv_normalized
            surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_normalized
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values.squeeze(-1), returns)

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
    """Convert a single observation dict (numpy arrays) to float tensors."""
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}


def train():
    # Hyperparameters
    BATCH_SIZE = 32768       # Collect this many steps before each PPO update
    LOG_INTERVAL = 10          # Print stats every N updates
    SAVE_INTERVAL = 50         # Save checkpoint every N updates
    ADVANCEMENT_THRESHOLD = 0.4  # Mean reward needed to advance curriculum
    ADVANCEMENT_WINDOW = 100   # Episodes to average over
    TOTAL_UPDATES = 10_000     # Total PPO updates to perform

    model = SpaceWarNet()
    trainer = PPOTrainer(
        model,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        epochs=4,
        minibatch_size=256,
        entropy_coef=0.01,
        vf_coef=0.5,
    )

    stage = 0
    reward_history = deque(maxlen=ADVANCEMENT_WINDOW)
    ep_reward = 0
    ep_count = 0
    update_count = 0

    buffer = RolloutBuffer()

    env = SW_1v1_env_singleplayer({"egocentric": True, "speed": 5.0, "ep_length": 4096, **curriculum_stages[stage]})
    obs, _ = env.reset()
    obs_tensor = tensorize_obs(obs[0])

    print(f"Starting training at stage {stage}: {curriculum_stages[stage]}")

    while update_count < TOTAL_UPDATES:
        # Collect rollout
        for _ in range(BATCH_SIZE):
            action, log_prob, value = trainer.select_action(obs_tensor)

            next_obs, rewards, term, trunc, info = env.step({0: action})
            reward = rewards[0]
            done = term["__all__"] or trunc["__all__"]

            buffer.add(obs_tensor, action, log_prob, value, reward, done)
            ep_reward += info[BASE_REWARD]

            if done:
                reward_history.append(ep_reward)
                ep_reward = 0
                ep_count += 1

                # Curriculum advancement check
                if len(reward_history) >= ADVANCEMENT_WINDOW:
                    mean_reward = np.mean(reward_history)
                    if mean_reward > ADVANCEMENT_THRESHOLD and stage < len(curriculum_stages) - 1:
                        stage += 1
                        reward_history.clear()
                        print(f"\n>>> Reward = {mean_reward:.3f}; Advancing to stage {stage}: {curriculum_stages[stage]}")

                # Reset with current curriculum stage
                env = SW_1v1_env_singleplayer({"egocentric": True, "speed": 5.0, 
                    "ep_length": 4096, **curriculum_stages[stage]})
                obs, _ = env.reset()
                obs_tensor = tensorize_obs(obs[0])
            else:
                obs = next_obs
                obs_tensor = tensorize_obs(obs[0])

        # PPO update
        if (reward_history): # MCW: Episodes are long; it needs this if statement or something more elegant.
            stats = trainer.update(buffer)
            update_count += 1

            # Logging
            if update_count % LOG_INTERVAL == 0:
                mean_r = np.mean(reward_history)
                n = stats["n_updates"] if stats["n_updates"] > 0 else 1
                print(
                    f"Update {update_count:5d} | "
                    f"Episodes {ep_count:6d} | "
                    f"Stage {stage} | "
                    f"Mean reward: {mean_r:+.3f} | "
                    f"Policy loss: {stats['policy_loss']/n:.4f} | "
                    f"Value loss: {stats['value_loss']/n:.4f} | "
                    f"Entropy: {stats['entropy']/n:.4f}"
                )

        # Save checkpoint
        if update_count % SAVE_INTERVAL == 0:
            path = f"checkpoints/spacewar_stage{stage}_update{update_count}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "stage": stage,
                "update_count": update_count,
                "ep_count": ep_count,
            }, path)
            print(f"  Saved checkpoint to {path}")


if __name__ == "__main__":
    import os
    os.makedirs("checkpoints", exist_ok=True)
    train()