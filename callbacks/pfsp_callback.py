from collections import defaultdict
from typing import Any, List, Dict
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.typing import EpisodeType

import numpy as np
import torch

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS
from ray.rllib.utils.annotations import override
from ray.rllib.core import (
    COMPONENT_RL_MODULE,
    COMPONENT_LEARNER,
    COMPONENT_LEARNER_GROUP,
)

from agent_comparison.BradleyTerry_rllib import get_BT_skill
from callbacks.elo_eval import print_elo_table
from callbacks.augment_critic_with_id import OPPONENT_ID

MAIN_MODULE = 'my_policy'
MAIN_EXPLOITER = 'main_exploiter'
APFSP_MODULES = [MAIN_MODULE, MAIN_EXPLOITER]
MAX_OPPONENTS = 200

# Get the agent that will be learning this episode, from the set of learning agents
def get_learning_agent(episode, policies_to_train):
  # Returns the agent (X or O) and the ID of the 'student' policy
  len_policies = len(policies_to_train)
  eid = hash(episode.id_) % (2*len_policies)
  agent_id = 1 if eid < len_policies else 0
  policy_id = policies_to_train[eid%len_policies]
  return agent_id, policy_id

### Log agent performance using Bradley-Terry
def build_wins(module_list, win_counts_t):
    wcs = np.zeros((len(module_list), len(module_list)))
    for i, a1 in enumerate(module_list):
        for j, a2 in enumerate(module_list):
            if (a1==a2):
                continue
            wcs[i][j] = win_counts_t[a1][a2] # How many times as a1 beaten a2?
    return wcs

### Probabilistic fictitious self play
def pfsp(agent, opponents, wr, rng):
    if (wr is None):
        return rng.choice(opponents)
    weights = np.array([wr[o][agent] for o in opponents])
    weights = weights**2 # Weight by square of opponent win rate
    wr_sum = weights.sum()
    if (wr_sum == 0):
        return rng.choice(opponents)
    return rng.choice(opponents, p=weights/wr_sum)

def create_atm_fn(agent_names, wr, just_added):
    ''' agent_names is a list of all league agents' names.
    wr is a dictionary of win rates '''
    def atm_fn(agent_id, episode, **kwargs):
        student_agent, student_policy = get_learning_agent(
            episode,
            APFSP_MODULES if MAIN_EXPLOITER in agent_names else [MAIN_MODULE],
        )
        eid = abs(hash(episode.id_))
        rng = np.random.default_rng(seed=eid)
        if (agent_id==student_agent):
            return student_policy
        if (student_policy == MAIN_MODULE):
            selector = rng.random()
            historical_names = list(filter(lambda s: (
                s not in APFSP_MODULES) and (s not in just_added), agent_names))
            if (selector < 0.5): # PFSP branch
                valid_options = historical_names # PFSP over all historical modules
            elif (selector < 0.65): # Verification branch; any historical agent with 70% WR
                if (wr is None):
                    return MAIN_MODULE
                valid_options = list(filter(lambda s: (wr[s][MAIN_MODULE] * 3/7 > wr[MAIN_MODULE][s]), historical_names))
            else: # Self-play branch
                return MAIN_MODULE 
            if (len(valid_options)==0):
                return MAIN_MODULE # Default to self-play
            return pfsp(MAIN_MODULE, valid_options, wr, rng)
        elif (student_policy == MAIN_EXPLOITER):
            return MAIN_MODULE # Main exploiter always plays against the main policy
        else:
            raise Exception(f"Unexpected student policy: `{student_policy}`")
    return atm_fn

def get_mc_string(agent1, agent2):
    return '-'.join(sorted([agent1, agent2]))

class PFSPCallback(RLlibCallback):
    def __init__(self, league_initial, module_name_to_id=None, clone_every=10, id_aug=False, warmup=0):
        super().__init__()
        self.clone_every = clone_every
        self.warmup = warmup # Extra steps to wait the first time around
        self.id_aug = id_aug # Does the main agent need ID embeddings updated when agents are cloned?
        self.module_name_to_id = module_name_to_id
        self.league = league_initial
        self.win_counts = defaultdict(lambda: defaultdict(lambda:0))
        self.match_counts = defaultdict(lambda:0)
        # Track previous totals to facilitate stat inheriting
        self.win_counts_t = defaultdict(lambda: defaultdict(lambda:0))
        self.match_counts_t = defaultdict(lambda:0)
        # Track previous best Bradley-Terry values
        self.prev_best = 0
        # Hacky fix for newly added modules not working properly
        self.just_added = []

    def on_sample_end(self, *, env_runner, metrics_logger, samples, **kwargs,) -> None:
        # This function is called by every env runner; results are collated afterwards.
        for episode in samples:
            if (episode.is_done):
                try:
                    rewards = episode.get_rewards()
                except Exception as e:
                    print("Exception while fetching rewards")
                    print(episode)
                    for aid, ep in episode.agent_episodes.items():
                        print(f" -- {aid}: {ep.module_id} -- ")
                        print(ep.actions)
                        print(ep.rewards)
                    continue
                assert len(rewards)==2
                w_agent, l_agent = list(rewards.keys())
                # Update stats
                outcome = np.sum(rewards[w_agent]) - np.sum(rewards[l_agent])
                if (outcome < 0):
                    w_agent, l_agent = l_agent, w_agent
                w_module = episode.module_for(w_agent)
                l_module = episode.module_for(l_agent)
                self.match_counts[get_mc_string(w_module, l_module)] += 1
                metrics_logger.log_value(
                    f"match_count_{get_mc_string(w_module, l_module)}",
                    1.0,
                    reduce='lifetime_sum'
                )
                if (outcome != 0):
                    metrics_logger.log_value(
                        f"win_count_{w_module}{l_module}",
                        0.5 if (w_module==l_module) else 1,
                        reduce='lifetime_sum'
                    )
                    
    def get_max_version(self, module_name):
        return max((int(s.split('_v')[-1]) for s in self.league if s.startswith(f"{module_name}_v")), default=0)

    def clone_agent(self, algorithm, to_clone):
        current_version = self.get_max_version(to_clone)
        prior_version_name = f"{to_clone}_v{current_version:03d}" if current_version != 0 else to_clone
        new_module_name = f"{to_clone}_v{current_version+1:03d}"
        self.just_added.append(new_module_name)
        self.league.append(new_module_name)
        print(f"adding new opponent to the mix ({new_module_name}).")
        for opponent in list(self.win_counts[to_clone].keys()): 
            # Reduce source match and win counts by x10 after cloning, to provide it with a 'fresh start'
            self.match_counts[get_mc_string(to_clone, opponent)] /= 10
            self.win_counts[to_clone][opponent] /= 10
            if (to_clone != opponent): # Avoid double-reduction
                self.win_counts[opponent][to_clone] /= 10
        # Add the new module
        cloned_module = algorithm.get_module(to_clone)
        algorithm.add_module(
            module_id=new_module_name,
            module_spec=RLModuleSpec.from_module(cloned_module),
        )
        module_updates = {new_module_name: cloned_module.get_state(),}
        # The embedding for the previous ID is copied into the new one whenever an update occurs.
        if (self.id_aug):
            # Main exploiter doesn't need its IDs updated because it only ever plays against main. We do this solely to future-proof.
            for learning_module_name in [MAIN_MODULE, MAIN_EXPLOITER]:
                if (learning_module_name not in self.league):
                    continue
                mm = algorithm.learner_group._learner._module[learning_module_name]
                critic_enc = mm.encoder.critic_encoder
                emb = critic_enc.embs[OPPONENT_ID] # The Embedding module for opponent identity, which we want to update.
                with torch.no_grad(): # Copy over id value from previous agent
                    emb.weight[self.module_name_to_id(new_module_name),:] = emb.weight[self.module_name_to_id(prior_version_name),:].clone()
                    print(f"UPDATED ENCODER EMBEDDING WEIGHTS AT INDEX {self.module_name_to_id(new_module_name)}: {emb.weight.shape} {emb.weight.sum(dim=1)[:5]}")
                module_updates[learning_module_name] = mm.get_state()
        # Syncs weights across everything
        algorithm.set_state({
            COMPONENT_LEARNER_GROUP: {
                COMPONENT_LEARNER: {
                    COMPONENT_RL_MODULE: module_updates
                }
            },
        })

    def build_stats_from_results(self, result):
        league_size = len(self.league)
        wrs = defaultdict(lambda:{})
        new_matches = {} # Just for debugging / observation
        for i in range(league_size-1):
            a = self.league[i]
            for j in range(i, league_size):
                b = self.league[j]
                # Record statistics involving a learning agent
                if (('_v' not in a) or ('_v' not in b)):
                    k = get_mc_string(a,b)
                    mc_total = result.get(f"match_count_{k}", 0)
                    wcab_total = result.get(f"win_count_{a}{b}", 0)
                    wcba_total = result.get(f"win_count_{b}{a}", 0)
                    new_matches[k] = mc_total - self.match_counts_t[k]
                    # Update match count tracker
                    self.match_counts[k] += new_matches[k]
                    self.match_counts_t[k] = mc_total # tracks previous result stat
                    # Update win count tracker
                    self.win_counts[a][b] += wcab_total - self.win_counts_t[a][b]
                    self.win_counts_t[a][b] = wcab_total
                    if (a!=b): # Don't double-count wins for self-play
                        self.win_counts[b][a] += wcba_total - self.win_counts_t[b][a]
                        self.win_counts_t[b][a] = wcba_total
                    if (self.match_counts[k] != 0):
                        wrs[a][b] = self.win_counts[a][b] / self.match_counts[k]
                        wrs[b][a] = self.win_counts[b][a] / self.match_counts[k]
                    else:
                        wrs[a][b] = 1/3
                        wrs[b][a] = 1/3 # Initialize as 33/33/33
        return wrs, new_matches

    def update_atm_fn(self, algorithm, wr):
        # Set new mapping function
        agent_to_module_mapping_fn = create_atm_fn(self.league, wr, self.just_added)
        algorithm.config._is_frozen = False
        algorithm.config.multi_agent(policy_mapping_fn=agent_to_module_mapping_fn)
        algorithm.config.freeze()
        # Add to (training) EnvRunners.
        def _add(_env_runner, _module_spec=None):
            _env_runner.config.multi_agent(
                policy_mapping_fn=agent_to_module_mapping_fn,
            )
            return MultiRLModuleSpec.from_module(_env_runner.module)
        algorithm.env_runner_group.foreach_env_runner(_add)

    def on_train_result(self, *, algorithm, metrics_logger=None, result, **kwargs):
        # This function is called only once, on a consistent worker.
        # Rebuild a set of stats with information from our env runners
        wrs, nm = self.build_stats_from_results(result[ENV_RUNNER_RESULTS])
        self.just_added = []
        iter = algorithm.iteration
        print(f"Iter={iter}:")
        print(f"Matchups: {dict(self.match_counts_t)}")
        print(f"Matchups (for probabilities): {dict(self.match_counts)}")
        print(f"Win rates (main):")
        for o, wr in wrs[MAIN_MODULE].items():
            new_matches = nm[get_mc_string(MAIN_MODULE,o)]
            if (new_matches!=0):
                wr_inv = wrs[o][MAIN_MODULE]
                dw = 1 - wr - wr_inv
                print(f'\t\t{o}: {wr:.02f}-{dw:.02f}-{wr_inv:.02f} (+{new_matches})')
        # Update and log BT scores (use the win counter that soft-resets)
        win_ar = build_wins(self.league, self.win_counts)
        bt_dict = get_BT_skill(self.league, win_ar)
        for k, v in bt_dict.items():
            algorithm.metrics.log_value(("BradleyTerry", k), v)
        bt_dict = algorithm.metrics.peek("BradleyTerry")
        print_elo_table(bt_dict)
        # Clone the agent if it's doing better than when it was last cloned or if it's beating the rest of the league
        if (len(self.league) < MAX_OPPONENTS) and ((iter)%(self.clone_every+self.warmup)==0):
            to_clone = set()
            threshold = min(self.prev_best, max(bt_dict.values()))
            if (bt_dict[MAIN_MODULE] >= threshold):
                to_clone.add(MAIN_MODULE)
                self.prev_best = bt_dict[MAIN_MODULE]
                self.warmup = 0
            # Main exploiter
            if MAIN_EXPLOITER in self.league:
                if (iter)%(2*self.clone_every)==0:
                    to_clone.add(MAIN_EXPLOITER)
                elif self.win_counts[MAIN_EXPLOITER][MAIN_MODULE] > (7/3)*self.win_counts[MAIN_MODULE][MAIN_EXPLOITER]:
                    to_clone.add(MAIN_EXPLOITER) # Clone the main exploiter if it has a 70% win rate against main policy
                    print(f'Cloning {MAIN_EXPLOITER} on merit')
            for m in to_clone:
                self.clone_agent(algorithm, m)
        # Update mapping function, reweighting and adding new module if needed
        self.update_atm_fn(algorithm, dict(wrs))