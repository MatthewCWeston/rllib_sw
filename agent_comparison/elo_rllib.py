# Takes a folder as an argument. Grabs every agent checkpoint in that folder, and runs evaluation.

import os
import sys
import importlib.util
import json
import torch
import functools
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from ray.rllib.utils.typing import ResultDict

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.tune.registry import register_env

from ray.rllib.utils.test_utils import add_rllib_example_script_args
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS
)

from classes.attention_encoder import AttentionPPOCatalog
from callbacks.checkpoint_restore_callback import LoadOnAlgoInitCallback
from callbacks.elo_eval import get_elo_rating, push_elo_rating, print_elo_table # import carryover components

from environments.SW_1v1_env import SW_1v1_env

# ==== Custom Evaluation Function ==== #

# Gets us an agent-to-module mapping function
def create_atm_fn(agent_names, w):
    ''' agent_names is a list of all league agents' names.
    w is a dictionary mapping each agent's name to its opponent probabilities. '''
    def atm_fn(agent_id, episode, **kwargs):
        eid = abs(hash(episode.id_))
        agent_ix = eid%len(agent_names)
        first = agent_names[agent_ix]
        if (agent_id==0): # For the first agent, pick randomly
            return first
        rng = np.random.default_rng(seed=abs(eid))
        return rng.choice(agent_names, p=w[first])
    return atm_fn
  
def get_weights(algorithm, agent_list, elo_ratings):
    '''ratings = np.array([elo_ratings], dtype=np.float64)
    diffs = np.abs(ratings - ratings.T)
    diffs -= diffs.min(axis=1, keepdims=True)
    diffs /= (diffs.max(axis=1, keepdims=True)+1e-6) # Normalize
    weights = np.exp(-(diffs)**2 / (2*diffs.var(axis=1, keepdims=True)+1e-6))'''
    weights = np.ones((len(agent_list), len(agent_list))) # No special weighting
    np.fill_diagonal(weights, 0) # No self-matches
    weights /= weights.sum(axis=1, keepdims=True) # Rows sum to 1
    return {a: weights[i] for i, a in enumerate(agent_list)}

def elo_eval_fn(
    algorithm: Algorithm,
    eval_workers: EnvRunnerGroup,
    agent_list: List[str], # Supplied when specifying custom_eval_fn
) -> Tuple[ResultDict, int, int]:
    elo_ratings = [get_elo_rating(algorithm, a, None) for a in agent_list]
    elo_dict = {a: [r, r] for a, r in zip(agent_list, elo_ratings)} # agent: (old, new)
    # Recalculate weights for each opponent based on ELO, apply custom eval function
    agent_to_module_mapping_fn = create_atm_fn(agent_list, get_weights(algorithm, agent_list, elo_ratings))
    '''algorithm.config._is_frozen = False
    algorithm.config.multi_agent(policy_mapping_fn=agent_to_module_mapping_fn)
    algorithm.config.freeze()'''
    def _add(_env_runner, _module_spec=None): # Add to (eval) EnvRunners.
        _env_runner.config.multi_agent(
            policy_mapping_fn=agent_to_module_mapping_fn,
        )
        return MultiRLModuleSpec.from_module(_env_runner.module)
    eval_workers.foreach_env_runner(_add)
    # Compile metrics; boilerplate stuff
    env_runner_metrics = []
    sampled_episodes = []
    eval_dur = algorithm.config.evaluation_duration
    episodes_and_metrics_all_env_runners = eval_workers.foreach_env_runner(
        # Return only the metrics, NOT the sampled episodes (we don't need them
        # anymore).
        func=lambda worker: (worker.sample(num_episodes=eval_dur), worker.get_metrics()),
        local_env_runner=False,
    )
    sampled_episodes.extend(
        eps
        for eps_and_mtrcs in episodes_and_metrics_all_env_runners
        for eps in eps_and_mtrcs[0]
    )
    env_runner_metrics.extend(
        eps_and_mtrcs[1] for eps_and_mtrcs in episodes_and_metrics_all_env_runners
    )
    algorithm.metrics.aggregate(
        env_runner_metrics, key=(EVALUATION_RESULTS, ENV_RUNNER_RESULTS)
    )
    eval_results = algorithm.metrics.peek((EVALUATION_RESULTS, ENV_RUNNER_RESULTS))
    env_steps = sum(eps.env_steps() for eps in sampled_episodes)
    agent_steps = sum(eps.agent_steps() for eps in sampled_episodes)
    # Okay, now go through the episodes and record winners, then update.
    for episode in sampled_episodes:
        rewards = episode.get_rewards()
        agents = list(rewards.keys())
        assert len(agents)==2
        score_diff = np.sum(rewards[agents[0]]) - np.sum(rewards[agents[1]])
        if (score_diff==0):
            continue # draw
        winner = 0 if score_diff > 0 else 1
        w_module = episode.module_for(agents[winner])
        l_module = episode.module_for(agents[(winner+1)%2])
        elo_w, elo_l = elo_dict[w_module][1], elo_dict[l_module][1]
        expected_w = 1 / (1 + 10 ** ((elo_l - elo_w)/400))
        # Update (K factor is just 20 for now; chess adjusts it slightly by player)
        adj = 20 * (1 - expected_w)
        elo_dict[w_module][1] += adj
        elo_dict[l_module][1] -= adj
    for o, (old, new) in elo_dict.items():
        push_elo_rating(algorithm, o, old, new)
    print_elo_table(algorithm.metrics.peek("ELO"))
    return eval_results, env_steps, agent_steps

# ====       Main Code Body       ==== #

# Handle arguments
parser = add_rllib_example_script_args(default_reward=40, default_iters=50)
parser.set_defaults(
    enable_new_api_stack=True,
    num_env_runners=0,
    evaluation_num_env_runners=1,
)
# Env 
parser.add_argument("--env-config", type=json.loads, default={})
# Architecture should match that of the base agent
parser.add_argument("--attn-dim", type=int, default=128) # Encoder dimensionality
parser.add_argument("--attn-ff-dim", type=int, default=2048) # Feedforward component of attention layers
parser.add_argument("--dropout", type=float, default=0.1) # In theory, this shouldn't help. In practice, it helps.
parser.add_argument("--activation-fn", type=str, default='relu') # Activation function for the network head.
parser.add_argument('--use-layernorm', action='store_true') # Use the norm
parser.add_argument('--fcnet', nargs='+', type=int, default=[256,256]) # Head architecture
# We'll need to load in a checkpoint, and it might be beneficial to cold-start the value function
parser.add_argument("--restore-checkpoint", type=os.path.abspath)
# Miscellaneous/logging
parser.add_argument("--render-every", type=int, default=0) # Every X steps, record a video
parser.add_argument("--elo-eval", action="store_true")
parser.add_argument("--agent-folder", type=os.path.abspath)
# Resources
parser.add_argument("--envs-per-env-runner", type=int, default=4)

args = parser.parse_args()

target_env = SW_1v1_env
register_env("env", lambda cfg: target_env(cfg))

# Configure run
callbacks = []
config = (
    PPOConfig()
    .environment(
        env="env",
        env_config=args.env_config,
    )
    .env_runners(
        num_cpus_per_env_runner=1,
        num_envs_per_env_runner=args.envs_per_env_runner,
    )
    .env_runners(
    num_env_runners=args.num_env_runners,
)
)
# Handle envs in MA format
env = target_env(args.env_config) # sample
agent_id = env.agents[0]
module_id = 'my_policy'
obs_space = env.observation_spaces[agent_id]
act_space = env.action_spaces[agent_id]
specs = {}

lrelu_override = args.activation_fn=="leakyrelu"
checkpoint_list = [f for f in Path(args.agent_folder).iterdir() if f.is_dir()]

agent_names = [x.name for x in checkpoint_list]
agent_weights = {}
for i, x in enumerate(agent_names):
    agent_weights[x] = np.ones(len(agent_names)) / (len(agent_names)-1)
    agent_weights[x][i] = 0
config.multi_agent(
    policies=agent_names,
    policy_mapping_fn=create_atm_fn(agent_names, agent_weights),
    policies_to_train=agent_names[:1], # Nothing's getting trained, of course, but it wants something here.
)

for checkpoint in checkpoint_list:
    print(f"Restoring checkpoint: {checkpoint.name}")
    specs[checkpoint.name] = RLModuleSpec(
        catalog_class=AttentionPPOCatalog,
        observation_space=obs_space,
        action_space=act_space,
        model_config={
            "attention_emb_dim": args.attn_dim,
            "attn_ff_dim": args.attn_ff_dim,
            "head_fcnet_hiddens": tuple(args.fcnet),
            "head_fcnet_activation": "relu" if lrelu_override else args.activation_fn,
            "override_activation_fn": lrelu_override,
            "vf_share_layers": False,
            "head_fcnet_use_layernorm": args.use_layernorm,
            "attn_layers": 1,
            "full_transformer": False,
            "recursive": False,
            "dropout": args.dropout,
        },
    )
    # Callback to load the respective checkpoint into this module on startup
    callbacks.append(functools.partial(
        LoadOnAlgoInitCallback,
        ckpt_path=checkpoint,
        module_name=module_id, # We're loading from a specific checkpoint path already.
        dest_module_names=[checkpoint.name], # Load weights into the curresponding agent
        vf_cold_start=False,
    ))
    
# TODO: Our evaluation loop should run standard sampling, then process the recorded episodes to track who won
# Use the example custom eval function, with the atm adjustment from my PFSP code.
# Episode postproc from my PFSP implementation as well.
#'''
assert args.evaluation_duration != "auto"
config.evaluation(
    evaluation_parallel_to_training=True,
    evaluation_num_env_runners=args.evaluation_num_env_runners,
    custom_evaluation_function=functools.partial(elo_eval_fn, agent_list=agent_names),
    evaluation_interval=1,  # How often to evaluate while training
    evaluation_duration=args.evaluation_duration, # Episodes to evaluate
) #'''

# Add spec
config.rl_module(
    rl_module_spec=MultiRLModuleSpec(
        rl_module_specs=specs
    ),
)
# Add callbacks
config.callbacks(callbacks)


# ====== RUN ======
# Run Evaluate() a bunch of times
algo = config.build_algo()
num_iters = 100

# Maybe print the difference between the before and after ELO scores (per episode?), and stop when it's low enough?
elo_old = defaultdict(lambda: 1200)
for i in range(num_iters):
  algo.evaluate()
  elo_new = algo.metrics.peek("ELO")
  total_diff = 0
  for k, v in elo_new.items():
      total_diff += np.abs(v - elo_old[k])
      elo_old[k] = elo_new[k]
  print(f'DIFFERENCE: {total_diff}')