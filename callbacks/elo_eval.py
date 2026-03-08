from collections import defaultdict
import numpy as np
import copy
import os
from pathlib import Path
from tabulate import tabulate

from typing import List, Tuple
from ray.rllib.utils.typing import ResultDict

from ray.rllib.core import (
    COMPONENT_ENV_RUNNER,
    COMPONENT_ENV_TO_MODULE_CONNECTOR,
    COMPONENT_MODULE_TO_ENV_CONNECTOR,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_LEARNER,
    COMPONENT_RL_MODULE,
    DEFAULT_MODULE_ID,
)
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.connectors.module_to_env import UnBatchToIndividualItems
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.env_runner import ENV_STEP_FAILURE
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.utils.metrics import (
    ENV_TO_MODULE_CONNECTOR,
    MODULE_TO_ENV_CONNECTOR,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_EPISODES,
    RLMODULE_INFERENCE_TIMER,
    WEIGHTS_SEQ_NO,
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS,
)
def get_opponent(episode_id, opponents, weights):
  rng = np.random.default_rng(np.abs(hash(episode_id)))
  return rng.choice(opponents, p=weights)

# Special sample implementation that loads a random opponent every epoch
def sample_elo(
        self,
        main_agent_name,
        checkpoint_list,
        opponent_weights=None,
        *,
        num_episodes,
    ) -> List[MultiAgentEpisode]:
        done_episodes_to_return: List[MultiAgentEpisode] = []
        env_ts = 0
        agent_ts = 0
        eps = 0
        # Have to reset the env (on all vector sub_envs).
        print(f'env: {self.env} num_envs: {self.num_envs} worker_index: {self.worker_index}')
        # when the number of env runners is not zero (no local), make_env doesn't seem to have been called here. Worker_index is zero. 
        self._reset_envs_and_episodes(False)
        self._needs_initial_reset = True
        if (num_episodes=='auto'):
          num_timesteps = (
            self.config.get_rollout_fragment_length(self.worker_index)
            * self.num_envs
          )
        # Instantiate opponents
        opponents = [RLModule.from_checkpoint( # Just grab the main module
            os.path.join(
                p,
                COMPONENT_LEARNER_GROUP,
                COMPONENT_LEARNER,
                COMPONENT_RL_MODULE,
                main_agent_name,
            )
        ) for p in checkpoint_list]
        if (len(opponents) > 0 and opponent_weights is None):
          opponent_weights = np.ones(len(opponents))/len(opponents)
        # Get the ID of the agent to swap out.
        agent_ids = list(self.env.envs[0].unwrapped.observation_spaces.keys())
        agent_ids.sort()
        main_agent_id = agent_ids[0]
        opponent_agent_id = agent_ids[-1]
        assert len(agent_ids) == 2 # This code is for competitive 2-player envs
        assert main_agent_id != opponent_agent_id
        # Instantiate an un-batcher for env-wise opponent processing
        unbatcher = UnBatchToIndividualItems()

        # Loop through `num_timesteps` timesteps or `num_episodes` episodes.
        while (
            (num_episodes=='auto' and env_ts < num_timesteps) or
            (num_episodes!='auto' and eps < num_episodes)
        ):
            # Env-to-module connector (already cached); from our reset above.
            to_module = self._cached_to_module
            assert to_module is not None
            self._cached_to_module = None
            if to_module:
                # MultiRLModule forward pass
                with self.metrics.log_time(RLMODULE_INFERENCE_TIMER):
                    # First, feed everything through the standard module.
                    to_module_copy = copy.deepcopy(to_module)
                    to_env = self.module.forward_inference(to_module)
                    # Second, go through and swap out the ones with the second ID
                    mms = self._shared_data['memorized_map_structure'][main_agent_name]
                    adi = to_env[main_agent_name][Columns.ACTION_DIST_INPUTS]
                    # Unbatch the observations
                    obs = unbatcher(rl_module=None,
                              batch=to_module_copy,
                              episodes=self._ongoing_episodes,
                              shared_data=self._shared_data,
                              )
                    obs = obs[main_agent_name][Columns.OBS]
                    for ix, (eid, aid) in enumerate(mms):
                        if (aid==opponent_agent_id and opponents):
                          assert len(obs[(eid, aid, main_agent_name)])==1 # one obs per eid/aid pair per timestep
                          o = obs[(eid, aid, main_agent_name)][0]
                          act = get_opponent(eid, opponents, opponent_weights).forward_inference({Columns.OBS: o})
                          adi[ix] = act[Columns.ACTION_DIST_INPUTS]
                    to_env = self._module_to_env(
                        rl_module=self.module,
                        batch=to_env,
                        episodes=self._ongoing_episodes,
                        explore=False,
                        shared_data=self._shared_data,
                        metrics=self.metrics,
                        metrics_prefix_key=(MODULE_TO_ENV_CONNECTOR,),
                    )
            else: # to_module == {}; we want to reset
                  to_env = {}
            self._shared_data["vector_env_episodes_map"] = {}
            # Extract the (vectorized) actions (to be sent to the env) from the module/connector output.
            actions = to_env.pop(Columns.ACTIONS, [{} for _ in self._ongoing_episodes])
            actions_for_env = to_env.pop(Columns.ACTIONS_FOR_ENV, actions)
            # Try stepping the environment.
            results = self._try_env_step(actions_for_env)
            # If the env step fails, reset the envs and continue the loop.
            if results == ENV_STEP_FAILURE:
                env_ts = 0
                agent_ts = 0
                self._reset_envs_and_episodes(False)
                continue
            observations, rewards, terminateds, truncateds, infos = results
            call_on_episode_start = set()
            # Store the data from the last environment step into the
            # episodes for all sub-environments.
            for env_index in range(self.num_envs):
                extra_model_outputs = defaultdict(dict)
                # `to_env` returns a dictionary with column keys and
                # (AgentID, value) tuple values.
                for col, ma_dict_list in to_env.items():
                    ma_dict = ma_dict_list[env_index]
                    for agent_id, val in ma_dict.items():
                        extra_model_outputs[agent_id][col] = val
                        extra_model_outputs[agent_id][
                            WEIGHTS_SEQ_NO
                        ] = self._weights_seq_no
                extra_model_outputs = dict(extra_model_outputs)
                # Episode has no data in it yet -> Was just reset and needs to be called with its `add_env_reset()` method.
                if not self._ongoing_episodes[env_index].is_reset:
                    # Add the reset step data to the episode.
                    self._ongoing_episodes[env_index].add_env_reset(
                        observations=observations[env_index],
                        infos=infos[env_index],
                    )
                    # Call the callback on episode start so users can hook in.
                    call_on_episode_start.add(env_index)
                # Call `add_env_step()` method on episode.
                else:
                    self._ongoing_episodes[env_index].add_env_step(
                        observations=observations[env_index],
                        actions=actions[env_index],
                        rewards=rewards[env_index],
                        infos=infos[env_index],
                        terminateds=terminateds[env_index],
                        truncateds=truncateds[env_index],
                        extra_model_outputs=extra_model_outputs,
                    )
                    # Ray metrics
                    self._log_env_steps(
                        metric=self._metrics_num_env_steps_sampled, num_steps=1
                    )
                    # Only increase ts when we actually stepped (not reset as a reset does not count as a timestep).
                    env_ts += self._increase_sampled_metrics(
                        1, observations[env_index], self._ongoing_episodes[env_index]
                    )
                    agent_ts += len(observations[env_index])

            done_episodes_to_run_env_to_module = []
            for env_index in range(self.num_envs):
                # Call `on_episode_start()` callback (always after reset).
                if env_index in call_on_episode_start:
                    self._make_on_episode_callback(
                        "on_episode_start", env_index, self._ongoing_episodes
                    )
                else: # Make the `on_episode_step` callbacks.
                    self._make_on_episode_callback(
                        "on_episode_step", env_index, self._ongoing_episodes
                    )
                if self._ongoing_episodes[env_index].is_done: # Episode is done.
                    eps += 1
                    if (len(checkpoint_list)!=0): # TEMP
                      # Record who won in the recorded episode's extra data
                      eix = self._ongoing_episodes[env_index]
                      rewards = eix.get_rewards()
                      eix.custom_data['opponent'] = get_opponent(eix.id_, checkpoint_list, opponent_weights)
                      eix.custom_data['score_diff'] = np.sum(rewards[main_agent_id]) - np.sum(rewards[opponent_agent_id])
                    self._make_on_episode_callback(
                        "on_episode_end", env_index, self._ongoing_episodes
                    )
                    self._prune_zero_len_sa_episodes(self._ongoing_episodes[env_index])
                    done_episodes_to_return.append(self._ongoing_episodes[env_index])
                    # Run a last time the `env_to_module` pipeline for these episodes to postprocess artifacts (e.g. observations to one-hot).
                    done_episodes_to_run_env_to_module.append(
                        self._ongoing_episodes[env_index]
                    )
                    old_episode_id = self._ongoing_episodes[env_index].id_
                    # Create a new episode object with no data in it.
                    self._new_episode(env_index, self._ongoing_episodes)
                    # Register the mapping of new episode ID to old episode ID.
                    self._shared_data["vector_env_episodes_map"].update(
                        {old_episode_id: self._ongoing_episodes[env_index].id_}
                    )
                    # Stop processing more envs if we've collected enough episodes.
                    if num_episodes != 'auto' and eps >= num_episodes:
                        break
            # Env-to-module connector pass (cache results as we will do the RLModule forward pass only in the next `while`-iteration).
            if self.module is not None:
                kwargs = {
                    Columns.OBS: observations,
                    Columns.ACTIONS: actions,
                    Columns.REWARDS: rewards,
                    Columns.INFOS: infos,
                    Columns.TERMINATEDS: terminateds,
                    Columns.TRUNCATEDS: truncateds,
                }
                if done_episodes_to_run_env_to_module:
                    # Run the env-to-module connector pipeline for all done episodes.
                    self._env_to_module(
                        batch={},
                        episodes=done_episodes_to_run_env_to_module,
                        explore=False,
                        rl_module=self.module,
                        shared_data=self._shared_data,
                        metrics=None,
                        **kwargs,
                    )
                self._cached_to_module = self._env_to_module(
                    episodes=self._ongoing_episodes,
                    batch={},
                    explore=False,
                    rl_module=self.module,
                    shared_data=self._shared_data,
                    metrics=self.metrics,
                    metrics_prefix_key=(ENV_TO_MODULE_CONNECTOR,),
                    **kwargs,
                )
        # Return done episodes
        self._done_episodes_for_metrics.extend(done_episodes_to_return)
        return done_episodes_to_return

def get_elo_rating(algorithm, agent_name, main_agent_name):
    try: # Get our stored ELO rating
      return algorithm.metrics.peek(("ELO", agent_name))
    except KeyError:
      if ('checkpoint_' not in agent_name): # Our default is 1200, except for checkpoints cloned from main
        algorithm.metrics.log_value(("ELO", agent_name),1200, reduce="lifetime_sum")
        return 1200
      main_elo = get_elo_rating(algorithm, main_agent_name, main_agent_name) # New checkpoints inherit from main
      algorithm.metrics.log_value(("ELO", agent_name), main_elo, reduce="lifetime_sum")
      return main_elo

def push_elo_rating(algorithm, agent_name, old, new):
    algorithm.metrics.log_value(("ELO", agent_name),new-old)

def update_elo(algorithm: Algorithm, sampled_episodes, main_agent_name):
  elo_main = old_elo_main = get_elo_rating(algorithm, main_agent_name, main_agent_name)
  elo_others = {}
  for e in sampled_episodes:
    # Get winner that we recorded in sample_elo
    opponent, score_diff = e.custom_data['opponent'], e.custom_data['score_diff']
    if (score_diff != 0):
      opponent = opponent.name
      if (opponent not in elo_others):
        opponent_elo = get_elo_rating(algorithm, opponent, main_agent_name)
        elo_others[opponent] = [opponent_elo, opponent_elo] # [old, new]
      elo_other = elo_others[opponent][1]
      # Get expected scores
      expected_main = 1 / (1 + 10 ** ((elo_other - elo_main)/400))
      # Update (K factor is just 20 for now; chess adjusts it slightly by player)
      adj = 20 * (1 - (score_diff < 0) - expected_main)
      elo_main += adj
      elo_others[opponent][1] -= adj
  push_elo_rating(algorithm, main_agent_name, old_elo_main, elo_main)
  for o, (old, new) in elo_others.items():
      push_elo_rating(algorithm, o, old, new)

def get_weights(algorithm, opponents, main_agent_name):
  if (len(opponents)==0):
    return None
  diffs = np.array([
        get_elo_rating(algorithm, o.name, main_agent_name) for o in opponents
    ]) - get_elo_rating(algorithm, main_agent_name, main_agent_name)
  diffs -= diffs.min()
  diffs /= (diffs.max()+1e-6) # Normalize
  weights = np.exp(-(diffs)**2 / (2*diffs.var()+1e-6))
  weights /= weights.sum()
  return weights

def print_elo_table(elo):
  sorted_keys = sorted([k for k in elo.keys()])
  table_data = [[k, elo[k]] for k in sorted_keys]
  print(tabulate(table_data, headers=["Checkpoint", "Rating"], tablefmt='rounded_outline', floatfmt=".0f"))

def elo_eval_fn(
    algorithm: Algorithm,
    eval_workers: EnvRunnerGroup,
    checkpoint_dir: str, # Supplied when specifying custom_eval_fn
    main_agent_name: str,
) -> Tuple[ResultDict, int, int]:
    # Gather up saved checkpoints
    checkpoint_list = [f for f in Path(checkpoint_dir).glob('checkpoint_*') if f.is_dir()]
    env_runner_metrics, sampled_episodes = [], []
    eval_episodes = algorithm.config.evaluation_duration # If not auto
    # Have the workers run a function that takes random main/earlier pairings and records the results
    '''
    episodes_and_metrics_all_env_runners = eval_workers.foreach_env_runner(
        # env: None num_envs: 0 worker_index: 0
        # (MultiAgentEnvRunner pid=5402) env: SyncVectorMultiAgentEnv(rllib-multi-agent-env-v0, num_envs=1) num_envs: 1 worker_index: 1
        func=lambda worker: print(f'env: {worker.env} num_envs: {worker.num_envs} worker_index: {worker.worker_index}'),
        local_env_runner=False,
    )
    #'''
    local_er = (algorithm.config.create_env_on_local_worker) or (algorithm.config.evaluation_num_env_runners==0)
    weights = get_weights(algorithm, checkpoint_list, main_agent_name)
    episodes_and_metrics_all_env_runners = eval_workers.foreach_env_runner(
        func=lambda worker: (sample_elo(
            worker,
            main_agent_name,
            checkpoint_list,
            weights,
            num_episodes=eval_episodes
        ), worker.get_metrics()),
        local_env_runner=local_er,
    ) #'''
    sampled_episodes.extend(
        eps
        for eps_and_mtrcs in episodes_and_metrics_all_env_runners
        for eps in eps_and_mtrcs[0]
    )
    env_runner_metrics.extend(
        eps_and_mtrcs[1] for eps_and_mtrcs in episodes_and_metrics_all_env_runners
    )
    # This will aggregate the metrics from the sampling into EVAlUATION_RESULTS
    algorithm.metrics.aggregate(
        env_runner_metrics, key=(EVALUATION_RESULTS, ENV_RUNNER_RESULTS)
    )
    eval_results = algorithm.metrics.peek((EVALUATION_RESULTS, ENV_RUNNER_RESULTS))
    # Aggregate sampled episodes
    env_steps = sum(eps.env_steps() for eps in sampled_episodes)
    agent_steps = sum(eps.agent_steps() for eps in sampled_episodes)
    # Sampled episodes will have (opponent, opponent_won) tuples in their auxiliary data. Update the ELO ratings accordingly.
    if (checkpoint_list):
      update_elo(algorithm, sampled_episodes, main_agent_name)
      print_elo_table(algorithm.metrics.peek("ELO"))
    return eval_results, env_steps, agent_steps