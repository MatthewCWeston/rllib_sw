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
from callbacks.elo_eval import print_elo_table # import carryover components

from environments.SW_1v1_env import SW_1v1_env

# ==== Custom Evaluation Function ==== #

# Gets us an agent-to-module mapping function
def create_atm_fn(module_list, w):
    ''' module_list is a list of all league agents' names.
    w is a dictionary mapping each module's name to its opponent probabilities. '''
    def atm_fn(agent_id, episode, **kwargs):
        eid = abs(hash(episode.id_))
        module_ix = eid%len(module_list)
        first = module_list[module_ix]
        if (agent_id==0): # For the first agent, pick randomly
            return first
        rng = np.random.default_rng(seed=abs(eid))
        return rng.choice(module_list, p=w[first])
    return atm_fn
  
def get_weights(algorithm, module_list):
    weights = np.ones((len(module_list), len(module_list))) # No special weighting
    np.fill_diagonal(weights, 0) # No self-matches
    weights /= weights.sum(axis=1, keepdims=True) # Rows sum to 1
    return {a: weights[i] for i, a in enumerate(module_list)}
    
def get_win_counts(algorithm, module_list):
    wcs = np.zeros((len(module_list), len(module_list)))
    for i, a1 in enumerate(module_list):
        for j, a2 in enumerate(module_list):
            if (a1==a2):
                continue
            try:
                wcs[i][j] = algorithm.metrics.peek(("WIN_COUNTS", a1, a2)) # How many times has a1 beaten a2?
            except KeyError:
                algorithm.metrics.log_value(("WIN_COUNTS", a1, a2), 0, reduce="lifetime_sum")
    return wcs
    
def get_BT_skill(module_list, wins, max_iter=1000, tol=1e-8): # Returns a name -> skill dictionary
    n = len(module_list)
    skill = np.ones(n)
    for i in range(max_iter):
        prev_skill = skill.copy()
        for ix in range(n): # Calculate skill for each module
            num, denom = 0, 1e-6
            for ox in range(n):
                st = skill[ox] + skill[ix] + 1e-6 # total skill
                num += wins[ix][ox] * skill[ox] / st # wins * opp_skill / skill_total
                denom += wins[ox][ix] / st # losses / skill_total
            skill[ix] = num / (denom+1e-6)
        skill /= skill[skill!=0].prod()**(1/n) # Normalize
        delta = np.linalg.norm(skill - prev_skill, ord=1)
        if delta < tol: # Check convergence
            print(f"Converged in {i+1} iterations.")
            return {a: s for a, s in zip(module_list, skill)}
    return {a: s for a, s in zip(module_list, skill)}

def elo_eval_fn(
    algorithm: Algorithm,
    eval_workers: EnvRunnerGroup,
    module_list: List[str], # Supplied when specifying custom_eval_fn
) -> Tuple[ResultDict, int, int]:
    # Gather and compile data
    # Compile metrics; boilerplate stuff
    env_runner_metrics = []
    sampled_episodes = []
    eval_dur = algorithm.config.evaluation_duration
    episodes_and_metrics_all_env_runners = eval_workers.foreach_env_runner(
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
    # Okay, now go through the episodes, record winners, update counts
    new_wins = defaultdict(lambda: defaultdict(lambda:0))
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
        new_wins[w_module][l_module] += 1
    for a1, wd in new_wins.items(): # Update metrics
        for a2, c in wd.items():
            algorithm.metrics.log_value(("WIN_COUNTS", a1, a2), c, reduce="lifetime_sum")
    wcs = get_win_counts(algorithm, module_list) # Load updated WRs into a dict
    skills = get_BT_skill(module_list, wcs)
    print_elo_table(skills)
    for k, v in skills.items():
        algorithm.metrics.log_value(("BradleyTerry", k), v)
    return eval_results, env_steps, agent_steps

# ====       Main Code Body       ==== #

if __name__ == '__main__':
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
    # Miscellaneous/logging
    parser.add_argument("--render-every", type=int, default=0) # Every X steps, record a video
    parser.add_argument("--elo-eval", action="store_true")
    parser.add_argument("--agent-folder", type=os.path.abspath)
    parser.add_argument("--eval-batch-size", type=int, default=50)
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
    checkpoint_list = [f for f in Path(args.agent_folder).iterdir() if f.is_dir() and f.name != 'wandb']

    module_list = [x.name for x in checkpoint_list]
    module_weights = {}
    for i, x in enumerate(module_list):
        module_weights[x] = np.ones(len(module_list)) / (len(module_list)-1)
        module_weights[x][i] = 0
    config.multi_agent(
        policies=module_list,
        policy_mapping_fn=create_atm_fn(module_list, module_weights),
        policies_to_train=module_list[:1], # Nothing's getting trained, of course, but it wants something here.
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
        if os.path.isdir(os.path.join(checkpoint, 'learner_group')): # Get main policy from each checkpoint
            callbacks.append(functools.partial(
                LoadOnAlgoInitCallback,
                ckpt_path=checkpoint,
                module_name=module_id, # We're loading from a specific checkpoint path already.
                dest_module_names=[checkpoint.name], # Load weights into the curresponding module
                vf_cold_start=False,
            ))
        else: # Get each agent from one checkpoint
            callbacks.append(functools.partial(
                LoadOnAlgoInitCallback,
                ckpt_path=str(checkpoint).split('learner_group')[0],
                module_name=checkpoint.name,
                dest_module_names=[checkpoint.name], # Load weights into the curresponding module
                vf_cold_start=False,
            ))
    # TODO: Our evaluation loop should run standard sampling, then process the recorded episodes to track who won
    # Use the example custom eval function, with the atm adjustment from my PFSP code.
    # Episode postproc from my PFSP implementation as well.
    #'''
    assert args.evaluation_duration != "auto"
    num_iters = args.evaluation_duration // args.eval_batch_size if args.evaluation_duration > args.eval_batch_size else 1
    config.evaluation(
        evaluation_parallel_to_training=True,
        evaluation_num_env_runners=args.evaluation_num_env_runners,
        custom_evaluation_function=functools.partial(elo_eval_fn, module_list=module_list),
        evaluation_interval=1,  # How often to evaluate while training
        evaluation_duration=args.evaluation_duration//num_iters, # Episodes to evaluate
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
    
    bt_old = defaultdict(lambda: 1)
    for i in range(num_iters):
        algo.evaluate() # Will print a table
        bt_new = algo.metrics.peek("BradleyTerry")
        total_diff = 0
        for k, v in bt_new.items():
            total_diff += np.abs(v - bt_old[k])
            bt_old[k] = bt_new[k]
        print(f'DIFFERENCE: {total_diff}')
