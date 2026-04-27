from collections import defaultdict

from typing import Any, Optional
from collections.abc import Callable
import gymnasium as gym
import numpy as np
import torch

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.connectors.connector_v2 import ConnectorV2
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

from callbacks.pfsp_agent import PFSPCallback, build_wins, pfsp, get_mc_string, MAIN_MODULE, MAX_OPPONENTS

MAIN_EXPLOITER = 'main_exploiter'
APFSP_MODULES = [MAIN_MODULE, MAIN_EXPLOITER]

import inspect

from callbacks.a_pfsp_callback import get_learning_agent

# Get the agent that will be learning this episode, from the set of learning agents
def get_learning_agent(episode, policies_to_train):
  # Returns the agent (X or O) and the ID of the 'student' policy
  len_policies = len(policies_to_train)
  eid = hash(episode.id_) % (2*len_policies)
  agent_id = 1 if eid < len_policies else 0
  policy_id = policies_to_train[eid%len_policies]
  return agent_id, policy_id

def create_atm_fn(agent_names, wr, just_added):
    ''' agent_names is a list of all league agents' names.
    wr is a dictionary of win rates '''
    def atm_fn(agent_id, episode, **kwargs):
        student_agent, student_policy = get_learning_agent(
            episode,
            APFSP_MODULES,
        )
        eid = abs(hash(episode.id_))
        rng = np.random.default_rng(seed=eid)
        if (eid % 2 == 0) != (agent_id==0):
            return student_policy
        if (student_policy == MAIN_MODULE):
            # Select an opponent.
            valid_options = list(filter(lambda s: s!=MAIN_MODULE and s not in just_added, agent_names))
            if (len(valid_options)==0): # Default to self-play
                return MAIN_MODULE
            return pfsp(MAIN_MODULE, valid_options, wr, rng)
        else:
            return MAIN_MODULE # Main exploiter always plays against the main policy
    return atm_fn

class APFSPCallback(PFSPCallback):
    
    @override(PFSPCallback)
    def clone_agent(self, algorithm, to_clone):
        # Clone an agent
        vid, increment = self.version_counter[to_clone], 1
        self.version_counter[to_clone] += 1
        new_module_id = f"{to_clone}_v{vid:03d}"
        if (to_clone==MAIN_EXPLOITER):
            vid, increment = (MAX_OPPONENTS-1) - vid, -1
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
        # TODO handle the addition of a main exploiter
        if (self.id_aug):
            mm = algorithm.learner_group._learner._module[MAIN_MODULE]
            critic_enc = mm.encoder.critic_encoder
            emb = critic_enc.embs[OPPONENT_ID] # The Embedding module for opponent identity, which we want to update.
            with torch.no_grad(): # Copy over id value from previous agent
                emb.weight[vid+increment,:] = emb.weight[vid,:].clone() # Main takes up index zero, our new agent's embedding is at vid+1. For Main Exploiter, we start at MAX_OPPONENTS and move downwards.
                print(f"UPDATED ENCODER EMBEDDING WEIGHTS AT INDEX {vid+increment}: {emb.weight.shape} {emb.weight.sum(dim=1)[:5]}")
            module_updates[MAIN_MODULE] = mm.get_state()
            # Main exploiter doesn't need its IDs updated because it only ever plays against main.
        # Syncs weights across everything
        algorithm.set_state({
            COMPONENT_LEARNER_GROUP: {
                COMPONENT_LEARNER: {
                    COMPONENT_RL_MODULE: module_updates
                }
            },
        })

    @override(PFSPCallback)
    def on_train_result(self, *, algorithm, metrics_logger=None, result, **kwargs):
        # This function is called only once, on a consistent worker.
        # Rebuild a set of stats with information from our env runners
        wrs, nm = self.build_stats_from_results(result[ENV_RUNNER_RESULTS])
        self.just_added = []
        iter = algorithm.iteration
        print(f"Iter={iter}:")
        print(f"Matchups: {dict(self.match_counts_t)}")
        print(f"Matchups (for probabilities): {dict(self.match_counts)}")
        print(f"Win rates ({MAIN_MODULE}):")
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
        if ((iter)%(self.clone_every+self.warmup)==0) and (len(self.league) < MAX_OPPONENTS):
            threshold = min(self.prev_best, max(bt_dict.values()))
            if (bt_dict[MAIN_MODULE] >= threshold):
                self.clone_agent(algorithm, MAIN_MODULE)
                self.prev_best = bt_dict[MAIN_MODULE]
                self.warmup = 0
            # Main exploiter
            if (iter)%(2*self.clone_every)==0:
                to_clone.add(MAIN_EXPLOITER)
        # Clone the main exploiter if it has an 70% win rate against the main agent.
        if self.win_counts[MAIN_EXPLOITER][MAIN_MODULE] >= (7/3)*self.win_counts[MAIN_MODULE][MAIN_EXPLOITER]:
            self.clone_agent(algorithm, MAIN_EXPLOITER)
            print(f'Cloning {MAIN_EXPLOITER} on merit')
        # Update mapping function, reweighting and adding new module if needed
        self.update_atm_fn(algorithm, dict(wrs))

# Disables learning for teacher agents' batches.
class DisableTeacherLearning(ConnectorV2):

    @override(ConnectorV2)
    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Dict[str, Any],
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        metrics: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> Any:
        for aid in batch:
            b_obs = batch[aid][Columns.OBS]
            if (Columns.LOSS_MASK not in batch[aid].keys()):
                batch[aid][Columns.LOSS_MASK] = torch.ones((b_obs.shape[0],), dtype=torch.bool).to(b_obs.device)
        start_indices = defaultdict(lambda: 0)
        lc = 0
        for mep in meps:
            student_agent, student_policy = get_learning_agent(mep, APFSP_MODULES)
            x_ep, o_ep = mep.agent_episodes['X'], mep.agent_episodes['O']
            x_mid, o_mid = x_ep.module_id, o_ep.module_id
            x_l, o_l = len(x_ep), len(o_ep)
            x_s = start_indices[x_mid]
            start_indices[x_mid]+=x_l
            o_s = start_indices[o_mid]
            start_indices[o_mid]+=o_l
            if (x_mid!=student_policy):
                batch[x_mid][Columns.LOSS_MASK][x_s:x_s+x_l] = False
            elif (o_mid!=student_policy):
                batch[o_mid][Columns.LOSS_MASK][o_s:o_s+o_l] = False