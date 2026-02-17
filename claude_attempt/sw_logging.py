# sw_logging.py
import time
import sys
import os
from collections import deque
import numpy as np


class TrainingDashboard:
    """Live-updating training dashboard for SpaceWar RL training."""

    SPARKLINE_WIDTH = 50  # Number of characters in the sparkline

    def __init__(self, total_updates, advancement_window=100, log_file="training_log.csv"):
        self.total_updates = total_updates
        self.advancement_window = advancement_window
        self.start_time = time.time()
        self.last_render_time = 0
        self.render_interval = 1.0

        # Tracking
        self.update_count = 0
        self.ep_count = 0
        self.stage = 0
        self.stage_config = {}

        # Reward tracking (rolling window for advancement decisions)
        self.base_reward_history = deque(maxlen=advancement_window)
        self.shaped_reward_history = deque(maxlen=advancement_window)

        # Full history for current stage (used for sparkline)
        self.stage_base_rewards = []
        self.stage_shaped_rewards = []

        # Per-update stats
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0
        self.last_entropy = 0.0

        # Stage history
        self.stage_transitions = []

        # CSV log
        self.log_file = log_file
        self._init_csv()

        # Event queue — stores (text, optional_sparkline) tuples
        self.events = deque(maxlen=20)

        # Rendering state
        self._last_line_count = 0
        self._first_render = True

        try:
            self.term_width = os.get_terminal_size().columns
        except OSError:
            self.term_width = 90

    def _init_csv(self):
        with open(self.log_file, "w") as f:
            f.write(
                "wall_time,update,episodes,stage,"
                "mean_base_reward,mean_shaped_reward,"
                "policy_loss,value_loss,entropy\n"
            )

    def _append_csv(self):
        elapsed = time.time() - self.start_time
        mean_base = np.mean(self.base_reward_history) if self.base_reward_history else 0.0
        mean_shaped = np.mean(self.shaped_reward_history) if self.shaped_reward_history else 0.0
        with open(self.log_file, "a") as f:
            f.write(
                f"{elapsed:.1f},{self.update_count},{self.ep_count},{self.stage},"
                f"{mean_base:.4f},{mean_shaped:.4f},"
                f"{self.last_policy_loss:.6f},{self.last_value_loss:.6f},{self.last_entropy:.4f}\n"
            )

    @staticmethod
    def _fmt_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _bar(self, value, max_value, width=30, fill="█", empty="░"):
        ratio = min(max(value / max_value, 0.0), 1.0) if max_value > 0 else 0.0
        filled = int(ratio * width)
        return fill * filled + empty * (width - filled)

    def _color(self, text, code):
        return f"\033[{code}m{text}\033[0m"

    def _reward_color(self, reward):
        if reward > 0.3:
            return self._color(f"{reward:+.3f}", "92")
        elif reward > 0.0:
            return self._color(f"{reward:+.3f}", "93")
        elif reward > -0.3:
            return self._color(f"{reward:+.3f}", "33")
        else:
            return self._color(f"{reward:+.3f}", "91")

    @staticmethod
    def _bucket_mean(values, num_buckets):
        """Partition values into num_buckets contiguous groups and return their means."""
        n = len(values)
        if n == 0:
            return []
        if n <= num_buckets:
            return list(values)
        arr = np.array(values)
        # Split indices as evenly as possible
        bucket_indices = np.array_split(np.arange(n), num_buckets)
        return [arr[idx].mean() for idx in bucket_indices]

    def _sparkline_colored(self, values, width=None):
        """Render a sparkline with per-character color based on value."""
        if width is None:
            width = self.SPARKLINE_WIDTH
        blocks = " ▁▂▃▄▅▆▇█"
        if not values:
            return ""
        bucketed = self._bucket_mean(values, width)
        mn, mx = -1.0, 1.0
        chars = []
        for v in bucketed:
            normalized = (v - mn) / (mx - mn + 1e-8)
            idx = int(min(max(normalized, 0.0), 1.0) * (len(blocks) - 1))
            block = blocks[idx]
            if v > 0.3:
                chars.append(self._color(block, "92"))
            elif v > 0.0:
                chars.append(self._color(block, "93"))
            elif v > -0.3:
                chars.append(self._color(block, "33"))
            else:
                chars.append(self._color(block, "91"))
        return "".join(chars)

    def log_episode(self, base_reward, shaped_reward):
        self.ep_count += 1
        self.base_reward_history.append(base_reward)
        self.shaped_reward_history.append(shaped_reward)
        self.stage_base_rewards.append(base_reward)
        self.stage_shaped_rewards.append(shaped_reward)
        self._maybe_render()

    def log_update(self, stats):
        self.update_count += 1
        n = max(stats.get("n_updates", 1), 1)
        self.last_policy_loss = stats["policy_loss"] / n
        self.last_value_loss = stats["value_loss"] / n
        self.last_entropy = stats["entropy"] / n
        self._append_csv()
        self._force_render()

    def log_stage_advance(self, new_stage, config, trigger_reward):
        elapsed = time.time() - self.start_time

        # Capture the completed stage's sparkline before clearing
        completed_stage = self.stage
        ep_count = len(self.stage_base_rewards)
        spark = self._sparkline_colored(self.stage_base_rewards)

        # Log the completion event with its graph
        self.events.appendleft((
            f"[{self._fmt_time(elapsed)}] "
            f"▲ Stage {completed_stage}→{new_stage} "
            f"(r={trigger_reward:+.3f}, {ep_count} eps)",
            f"    {spark}"
        ))

        # Update stage
        self.stage = new_stage
        self.stage_config = config
        self.stage_transitions.append((elapsed, new_stage, trigger_reward))

        # Clear stage-specific history
        self.stage_base_rewards = []
        self.stage_shaped_rewards = []

        self._force_render()

    def log_save(self, path):
        elapsed = time.time() - self.start_time
        self.events.appendleft((
            f"[{self._fmt_time(elapsed)}] Saved: {os.path.basename(path)}",
            None
        ))
        self._force_render()

    def log_event(self, message):
        elapsed = time.time() - self.start_time
        self.events.appendleft((f"[{self._fmt_time(elapsed)}] {message}", None))
        self._force_render()

    def set_stage(self, stage, config):
        self.stage = stage
        self.stage_config = config

    def _maybe_render(self):
        now = time.time()
        if now - self.last_render_time >= self.render_interval:
            self._force_render()

    def _force_render(self):
        self.last_render_time = time.time()
        self._render()

    def _build_lines(self):
        elapsed = time.time() - self.start_time
        mean_base = np.mean(self.base_reward_history) if self.base_reward_history else 0.0
        mean_shaped = np.mean(self.shaped_reward_history) if self.shaped_reward_history else 0.0
        progress = self.update_count / self.total_updates if self.total_updates > 0 else 0.0

        if progress > 0 and self.update_count > 0:
            eta_str = self._fmt_time(elapsed / progress - elapsed)
        else:
            eta_str = "--:--:--"

        w = min(self.term_width, 90)
        sep = lambda label: self._color(f"─── {label} ", "36") + "─" * max(0, w - len(label) - 5)

        lines = []
        lines.append("")
        lines.append(self._color("═" * w, "36"))
        lines.append(self._color("SPACEWAR RL TRAINING".center(w), "1;36"))
        lines.append(self._color("═" * w, "36"))
        lines.append("")

        # Time & Progress
        lines.append(sep("Time & Progress"))
        prog_bar = self._bar(self.update_count, self.total_updates, width=35)
        lines.append(
            f"  Elapsed: {self._fmt_time(elapsed)}  │  "
            f"ETA: {eta_str}  │  "
            f"Updates: {self.update_count}/{self.total_updates}"
        )
        lines.append(f"  [{prog_bar}] {progress * 100:5.1f}%")
        lines.append("")

        # Curriculum
        lines.append(sep("Curriculum"))
        stage_bar = self._bar(self.stage, 7, width=16, fill="■", empty="·")
        lines.append(f"  Stage: {self.stage}/7 [{stage_bar}]")
        cfg = self.stage_config
        if cfg:
            lines.append(
                f"  grav={cfg.get('grav_multiplier', '?'):.1f}  "
                f"size={cfg.get('size_multiplier', '?'):.1f}  "
                f"speed={cfg.get('target_speed', '?'):.1f}  "
                f"ammo={cfg.get('target_ammo', '?'):.1f}"
            )
        lines.append("")

        # Rewards
        lines.append(sep("Rewards"))
        ep_window = len(self.base_reward_history)
        base_col = self._reward_color(mean_base)
        shaped_col = self._reward_color(mean_shaped)
        lines.append(
            f"  Base reward (mean/{ep_window}):   {base_col}    │  "
            f"Episodes: {self.ep_count}"
        )
        lines.append(
            f"  Shaped reward (mean/{ep_window}): {shaped_col}"
        )

        # Stage progression sparkline
        stage_ep_count = len(self.stage_base_rewards)
        if stage_ep_count >= 2:
            spark = self._sparkline_colored(self.stage_base_rewards)
            lines.append(f"  Stage {self.stage} ({stage_ep_count} eps): {spark}")
        elif stage_ep_count > 0:
            lines.append(f"  Stage {self.stage} ({stage_ep_count} eps): (collecting...)")
        lines.append("")

        # PPO Stats
        lines.append(sep("PPO Stats"))
        lines.append(
            f"  π loss: {self.last_policy_loss:+.5f}  │  "
            f"V loss: {self.last_value_loss:.5f}  │  "
            f"Entropy: {self.last_entropy:.4f}"
        )
        lines.append("")

        # Event Log
        lines.append(sep("Event Log"))
        max_events = 10
        event_count = 0
        if self.events:
            for text, sparkline in list(self.events):
                if event_count >= max_events:
                    break
                lines.append(f"  {text}")
                event_count += 1
                if sparkline is not None and event_count < max_events:
                    lines.append(f"  {sparkline}")
                    event_count += 1
        else:
            lines.append("  (no events yet)")
        lines.append("")

        return lines

    def _render(self):
        lines = self._build_lines()
        new_line_count = len(lines)

        if self._first_render:
            self._first_render = False
        else:
            sys.stdout.write(f"\033[{self._last_line_count}A")
            sys.stdout.write("\033[J")

        sys.stdout.write("\n".join(lines))
        sys.stdout.write("\n")
        sys.stdout.flush()

        self._last_line_count = new_line_count