# sw_main.py
import numpy as np
import torch
import os
import multiprocessing as mp
from multiprocessing import shared_memory

from sw_logging import TrainingDashboard
from sw_ppo import RolloutBuffer, PPOTrainer
from environments.SpaceWar_constants import NUM_MISSILES
from sw_env import BASE_REWARD

def make_env(stage, target_env, curriculum):
    return target_env({
        "egocentric": True,
        "speed": 5.0,
        "ep_length": 4096,
        **curriculum[stage],
    })
    
def get_env_constants(target_env, curriculum):
    env = make_env(0, target_env, curriculum)
    obs_shapes = { k: v.shape for k, v in env.reset()[0][0].items() }
    obs_size = sum(np.prod(s) for s in obs_shapes.values()) # total floats for one obs
    act_count = env.action_spaces[0].shape[0] # Number of MultiDiscrete actions taken
    return obs_shapes, obs_size, act_count

def obs_dict_to_numpy(obs_dict, obs_shapes):
    """Flatten an obs dict into a 1-D float32 array matching obs_shapes order."""
    parts = []
    for key in obs_shapes:
        parts.append(np.asarray(obs_dict[key], dtype=np.float32).ravel())
    return np.concatenate(parts)


def numpy_to_obs_dict(arr, obs_shapes):
    """Reconstruct a {key: tensor} dict from a flat float32 array."""
    obs = {}
    offset = 0
    for key, shape in obs_shapes.items():
        size = int(np.prod(shape))
        obs[key] = torch.from_numpy(arr[offset : offset + size].reshape(shape).copy())
        offset += size
    return obs


# ─── Worker ──────────────────────────────────────────────────────────────────

def env_worker(
    worker_id: int,
    initial_stage: int,
    target_env,
    curriculum,
    obs_shapes, obs_size, act_count:int, result_size:int,
    result_shm_name: str,   # shared memory block the worker WRITES obs/reward into
    action_shm_name: str,   # shared memory block the worker READS actions from
    action_event: mp.Event, # main  -> worker: "action is ready"
    result_event: mp.Event, # worker -> main:  "result is ready"
    close_event:  mp.Event, # main  -> worker: "please exit"
):
    # ── Attach to shared memory ───────────────────────────────────────────
    res_shm = shared_memory.SharedMemory(name=result_shm_name)
    act_shm = shared_memory.SharedMemory(name=action_shm_name)

    # Numpy views directly into the shared buffers (zero-copy)
    result_arr = np.ndarray((result_size,),  dtype=np.float32, buffer=res_shm.buf)
    action_arr = np.ndarray((act_count,),            dtype=np.int32,
                             buffer=act_shm.buf,
                             offset=worker_id * act_count * np.dtype(np.int32).itemsize)

    def write_result(obs_flat, reward, done, base_reward):
        result_arr[:obs_size]    = obs_flat
        result_arr[obs_size]     = reward
        result_arr[obs_size + 1] = float(done)
        result_arr[obs_size + 2] = base_reward

    # ── Boot env ──────────────────────────────────────────────────────────
    stage = initial_stage
    env   = make_env(stage, target_env, curriculum)
    raw_obs, _ = env.reset()
    write_result(obs_dict_to_numpy(raw_obs[0], obs_shapes), 0.0, False, 0.0)
    result_event.set()   # signal: initial obs ready

    # ── Main loop ─────────────────────────────────────────────────────────
    while True:
        action_event.wait()
        action_event.clear()

        if close_event.is_set():
            res_shm.close()
            act_shm.close()
            return

        action = action_arr.copy()  # copy before main process can overwrite

        cmd = int(action[0])

        # ── Stage-change sentinel: action = [-2, new_stage, 0] ───────────
        if cmd == -2:
            stage = int(action[1])
            env   = make_env(stage, target_env, curriculum)
            raw_obs, _ = env.reset()
            write_result(obs_dict_to_numpy(raw_obs[0], obs_shapes), 0.0, False, 0.0)
            result_event.set()
            continue

        # ── Normal step ───────────────────────────────────────────────────
        next_raw, rewards, term, trunc, info = env.step({0: action})
        reward      = float(rewards[0])
        done        = term["__all__"] or trunc["__all__"]
        base_reward = float(info.get(BASE_REWARD, reward))

        if done:
            # Reuse existing env object - just reset it
            next_raw, _ = env.reset()

        write_result(obs_dict_to_numpy(next_raw[0], obs_shapes), reward, done, base_reward)
        result_event.set()


# ─── Manager ─────────────────────────────────────────────────────────────────

class SharedMemoryEnvManager:
    """
    Owns N env workers.  Communication is via shared memory; only tiny
    synchronisation signals (Events) cross the process boundary.
    """

    def __init__(self, num_envs: int, stage: int, target_env, curriculum):
        self.num_envs = num_envs
        self.obs_shapes, self.obs_size, act_count = get_env_constants(target_env, curriculum)
        result_size = self.obs_size + 3 # Shared result layout per worker: [obs...] [reward] [done] [base_reward]

        # ── Allocate one result block per worker ──────────────────────────
        self._res_shms: list[shared_memory.SharedMemory] = []
        self._res_arrs: list[np.ndarray]                 = []
        for _ in range(num_envs):
            shm = shared_memory.SharedMemory(
                create=True, size=int(result_size * np.dtype(np.float32).itemsize)
            )
            arr = np.ndarray((result_size,), dtype=np.float32, buffer=shm.buf)
            self._res_shms.append(shm)
            self._res_arrs.append(arr)

        # ── Single shared action block for all workers ────────────────────
        # Layout: worker_i reads int32[i*3 : i*3+3]
        self._act_shm = shared_memory.SharedMemory(
            create=True, size=int(num_envs * act_count * np.dtype(np.int32).itemsize)
        )
        self._act_arr = np.ndarray(
            (num_envs, act_count), dtype=np.int32, buffer=self._act_shm.buf
        )

        # ── Per-worker synchronisation events ─────────────────────────────
        self._action_events = [mp.Event() for _ in range(num_envs)]
        self._result_events = [mp.Event() for _ in range(num_envs)]
        self._close_event   = mp.Event()   # shared; set once to stop all workers

        # ── Launch workers ────────────────────────────────────────────────
        self.processes: list[mp.Process] = []
        for i in range(num_envs):
            p = mp.Process(
                target=env_worker,
                args=(
                    i, stage,
                    target_env, curriculum,
                    self.obs_shapes, self.obs_size, act_count, result_size,
                    self._res_shms[i].name,
                    self._act_shm.name,
                    self._action_events[i],
                    self._result_events[i],
                    self._close_event,
                ),
                daemon=True,
            )
            p.start()
            self.processes.append(p)

        # ── Wait for all workers to write their initial obs ───────────────
        self.current_obs: list[dict] = [None] * num_envs
        for i in range(num_envs):
            self._result_events[i].wait()
            self._result_events[i].clear()
            self.current_obs[i] = numpy_to_obs_dict(self._res_arrs[i][:self.obs_size], self.obs_shapes)

    # ── Inference helpers ─────────────────────────────────────────────────

    def get_batched_obs(self) -> dict[str, torch.Tensor]:
        """Stack current obs across all envs - used for batched inference."""
        batched = {}
        for key in self.obs_shapes:
            batched[key] = torch.stack([o[key] for o in self.current_obs], dim=0)
        return batched

    # ── Step ──────────────────────────────────────────────────────────────

    def step_async(self, actions_list):
        """
        Write all actions into shared memory, then signal every worker
        simultaneously.  Non-blocking from the main process perspective.
        """
        for i, action in enumerate(actions_list):
            self._act_arr[i] = action          # write into shared buffer
        # Signal after ALL writes so workers never see a partial state
        for i in range(self.num_envs):
            self._action_events[i].set()

    def step_wait(self):
        """
        Block until every worker has written its result, then read results.
        Returns list of (obs_dict, reward, done, base_reward).
        """
        results = []
        for i in range(self.num_envs):
            self._result_events[i].wait()
            self._result_events[i].clear()

            arr         = self._res_arrs[i]
            obs         = numpy_to_obs_dict(arr[:self.obs_size], self.obs_shapes)
            reward      = float(arr[self.obs_size])
            done        = bool(arr[self.obs_size + 1])
            base_reward = float(arr[self.obs_size + 2])

            self.current_obs[i] = obs
            results.append((obs, reward, done, base_reward))
        return results

    # ── Curriculum stage change ───────────────────────────────────────────

    def set_stage(self, stage: int):
        """Tell all workers to rebuild their env at the new curriculum stage."""
        for i in range(self.num_envs):
            self._act_arr[i][:2] = [-2, stage]   # sentinel
        for i in range(self.num_envs):
            self._action_events[i].set()
        for i in range(self.num_envs):
            self._result_events[i].wait()
            self._result_events[i].clear()
            self.current_obs[i] = numpy_to_obs_dict(self._res_arrs[i][:self.obs_size], self.obs_shapes)

    # ── Teardown ──────────────────────────────────────────────────────────

    def close(self):
        self._close_event.set()
        for i in range(self.num_envs):
            self._act_arr[i][:2] = [-1, 0]       # wake workers so they check close_event
            self._action_events[i].set()
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        for shm in self._res_shms:
            shm.close()
            shm.unlink()
        self._act_shm.close()
        self._act_shm.unlink()


# ─── Main training loop ───────────────────────────────────────────────────────

def train(num_workers, args, model, target_env, curriculum):
    BATCH_SIZE           = args.batch_size
    SAVE_INTERVAL        = 50
    ADVANCEMENT_THRESHOLD = 0.4
    ADVANCEMENT_WINDOW   = 100
    TOTAL_UPDATES        = 10_000

    os.makedirs("checkpoints", exist_ok=True)

    model = model(hidden=args.h_dim, emb=args.emb)
    trainer = PPOTrainer(
        model, lr=args.lr, gamma=args.gamma, lam=args.lam,
        clip_eps=0.2, epochs=4, minibatch_size=args.minibatch_size,
        entropy_coef=0.01, vf_coef=0.5,
    )

    stage = 0
    dash  = TrainingDashboard(
        total_updates=TOTAL_UPDATES,
        advancement_window=ADVANCEMENT_WINDOW,
    )
    dash.set_stage(stage, curriculum[stage])
    dash.log_event(f"Starting training - stage 0 with {num_workers} workers")

    env_mgr      = SharedMemoryEnvManager(num_workers, stage, target_env, curriculum)
    update_count = 0

    buffers          = [RolloutBuffer() for _ in range(num_workers)]
    ep_base_rewards  = [0.0] * num_workers
    ep_shaped_rewards= [0.0] * num_workers

    while update_count < TOTAL_UPDATES:
        steps_collected = 0

        while steps_collected < BATCH_SIZE:
            # ── Batched inference ─────────────────────────────────────────
            obs_batch = env_mgr.get_batched_obs()
            actions_list, log_probs_list, values_list = trainer.select_action_batch(obs_batch)

            # Snapshot obs references before the step overwrites current_obs
            pre_step_obs = [env_mgr.current_obs[i] for i in range(num_workers)]

            # ── Async step ────────────────────────────────────────────────
            env_mgr.step_async(actions_list)
            results = env_mgr.step_wait()

            # ── Store transitions ─────────────────────────────────────────
            for i, (obs_tensor, reward, done, base_reward) in enumerate(results):
                buffers[i].add(
                    pre_step_obs[i],
                    actions_list[i],
                    log_probs_list[i],
                    values_list[i],
                    reward,
                    done,
                )
                ep_base_rewards[i]   += base_reward
                ep_shaped_rewards[i] += reward

                if done:
                    dash.log_episode(ep_base_rewards[i], ep_shaped_rewards[i])
                    ep_base_rewards[i]   = 0.0
                    ep_shaped_rewards[i] = 0.0

            steps_collected += num_workers

        # ── Curriculum check ──────────────────────────────────────────────
        if len(dash.base_reward_history) >= ADVANCEMENT_WINDOW:
            mean_base = np.mean(dash.base_reward_history)
            if mean_base > ADVANCEMENT_THRESHOLD and stage < len(curriculum) - 1:
                stage += 1
                dash.base_reward_history.clear()
                dash.shaped_reward_history.clear()
                dash.log_stage_advance(stage, curriculum[stage], mean_base)
                env_mgr.set_stage(stage)
                for b in buffers:
                    b.clear()
                ep_base_rewards  = [0.0] * num_workers
                ep_shaped_rewards= [0.0] * num_workers
                continue

        # ── PPO update ────────────────────────────────────────────────────
        stats = trainer.update(buffers)
        if stats is None:
            continue

        update_count += 1
        dash.log_update(stats)

        if update_count % SAVE_INTERVAL == 0:
            path = f"checkpoints/spacewar_stage{stage}_update{update_count}.pt"
            torch.save({
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "stage":                stage,
                "update_count":         update_count,
                "ep_count":             dash.ep_count,
            }, path)
            dash.log_save(path)

    env_mgr.close()
    dash.log_event("Training complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    parser.add_argument("--batch-size",  type=int, default=32768)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--lam", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=3e-4) 
    parser.add_argument("--tl", action="store_true")
    
    parser.add_argument("--emb", type=int, default=64)
    parser.add_argument("--h-dim", type=int, default=256)
    
    args = parser.parse_args()
    
    if (args.tl): # Use the target-leading test environment 
        from target_leading_test.sw_curriculum_tl import curriculum_stages_tl
        from target_leading_test.sw_agent_tl import SpaceWarNet_Attention
        from target_leading_test.sw_env_tl import SW_lead_target
        target_env = SW_lead_target
        curriculum = curriculum_stages_tl
        model = SpaceWarNet_Attention
    else:        # Use the target environment
        from sw_curriculum import curriculum_stages
        from sw_agent import SpaceWarNet
        from sw_env import SW_1v1_env_singleplayer
        target_env = SW_1v1_env_singleplayer
        curriculum = curriculum_stages
        model = SpaceWarNet

    mp.set_start_method("spawn", force=True)   # required for shared_memory on all platforms
    train(num_workers=args.num_workers, args=args, model=model, target_env=target_env, curriculum=curriculum)