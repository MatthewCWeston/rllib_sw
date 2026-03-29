from collections import defaultdict

import numpy as np
import torch

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS
from ray.rllib.core import (
    COMPONENT_RL_MODULE,
    COMPONENT_LEARNER,
    COMPONENT_LEARNER_GROUP,
)

from agent_comparison.BradleyTerry_rllib import get_BT_skill
from callbacks.elo_eval import print_elo_table
from callbacks.augment_critic_with_id import OPPONENT_ID

MAIN_MODULE = 'my_policy'
MAX_OPPONENTS = 50

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
        eid = abs(hash(episode.id_))
        rng = np.random.default_rng(seed=eid)
        if (eid % 2 == 0) != (agent_id==0):
            return MAIN_MODULE
        # Select an opponent.
        valid_options = list(filter(lambda s: s!=MAIN_MODULE and s not in just_added, agent_names))
        if (len(valid_options)==0):
            return MAIN_MODULE
        return pfsp(MAIN_MODULE, valid_options, wr, rng)
    return atm_fn

def get_mc_string(agent1, agent2):
    return '-'.join(sorted([agent1, agent2]))

class PFSPCallback(RLlibCallback):
    def __init__(self, league_initial, clone_every=10, id_aug=False, warmup=0):
        super().__init__()
        self.clone_every = clone_every
        self.warmup = warmup # Extra steps to wait the first time around
        self.id_aug = id_aug # Does the main agent need ID embeddings updated when agents are cloned?
        self.league = league_initial
        self.win_counts = defaultdict(lambda: defaultdict(lambda:0))
        self.match_counts = defaultdict(lambda:0)
        self.version_counter = defaultdict(lambda: 1)
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
                rewards = episode.get_rewards()
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

    def clone_agent(self, algorithm, to_clone):
        # Clone an agent
        vid = self.version_counter[to_clone]
        self.version_counter[to_clone] += 1
        new_module_id = f"{to_clone}_v{vid}"
        self.just_added.append(new_module_id)
        self.league.append(new_module_id)
        print(f"adding new opponent to the mix ({new_module_id}).")
        for opponent in list(self.win_counts[to_clone].keys()): 
            # Reduce source match and win counts by x10 after cloning, to provide it with a 'fresh start'
            self.match_counts[get_mc_string(to_clone, opponent)] /= 10
            self.win_counts[to_clone][opponent] /= 10
            if (to_clone != opponent): # Avoid double-reduction
                self.win_counts[opponent][to_clone] /= 10
        # Add the new module
        cloned_module = algorithm.get_module(to_clone)
        algorithm.add_module(
            module_id=new_module_id,
            module_spec=RLModuleSpec.from_module(cloned_module),
        )
        module_updates = {new_module_id: cloned_module.get_state(),}
        # The embedding for the previous ID is copied into the new one whenever an update occurs.
        if (self.id_aug):
            mm = algorithm.learner_group._learner._module[MAIN_MODULE]
            critic_enc = mm.encoder.critic_encoder
            emb = critic_enc.embs[OPPONENT_ID] # The Embedding module for opponent identity, which we want to update.
            with torch.no_grad(): # Copy over id value from previous agent
                emb.weight[vid+1,:] = emb.weight[vid,:].clone() # Main takes up index zero, our new agent's embedding is at vid+1
                print(f"UPDATED ENCODER EMBEDDING WEIGHTS AT INDEX {vid+1}: {emb.weight.shape} {emb.weight.sum(dim=1)[:5]}")
            module_updates[MAIN_MODULE] = mm.get_state()
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
        # Clone the agent if it's doing better than its previous best
        if ((iter)%(self.clone_every+self.warmup)==0) and (bt_dict[MAIN_MODULE] > self.prev_best) and (len(self.league) < MAX_OPPONENTS):
            self.clone_agent(algorithm, MAIN_MODULE)
            self.prev_best = bt_dict[MAIN_MODULE]
            self.warmup = 0
        # Update mapping function, reweighting and adding new module if needed
        self.update_atm_fn(algorithm, dict(wrs))