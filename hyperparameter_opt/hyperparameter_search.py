
from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import ray

from ray import tune, train
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import register_env
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter

from environments.SW_1v1_env_singleplayer import SW_1v1_env_singleplayer
from classes.attention_encoder import AttentionPPOCatalog
from classes.batched_critic_ppo import BatchedCriticPPOLearner
from hyperparameter_opt.progress_reporter import CustomReporter

if TYPE_CHECKING:
    from ray.tune.experiment import Trial

PROJECT_PATH = Path(__file__).parent

# python hyperparameter_search.py --num-env-runners=2 --num-concurrent=5 --stop-threshold=1.0
# python hyperparameter_search.py --num-env-runners=5 --num-concurrent=10 --time-budget=5.0

# python hyperparameter_opt/hyperparameter_analysis.py --experiment-name "Test_Hyperparameter_Search"

parser = argparse.ArgumentParser()
parser.add_argument("--num-env-runners", type=int, default=2) # Per-trial, not total
parser.add_argument("--num-concurrent", type=int, default=5) # Per-trial, not total
parser.add_argument("--stop-threshold", type=float, default=float('inf'))
parser.add_argument("--time-budget", type=float, default=1.5) # In hours, total
args = parser.parse_args()

"""
Performs a hyperparameter search.
Multiple trials execute in parallel and training progress is
compared to prune under-performing trials.
"""

experiment_name = f'Test_Hyperparameter_Search'
max_iter_individual = 50
max_time_total = int(3600 * args.time_budget)
grace_period_iter = 7

# Register our Environment and Create the Config Object
target_env = SW_1v1_env_singleplayer
register_env("env", lambda cfg: target_env(cfg))
env_cfg = {"speed": 5.0, "ep_length": 4096, 'grav_multiplier': 0.1, 'egocentric':True}
module_id = 'my_policy'
env = target_env() # sample
agent_id = env.agents[0]
obs_space = env.observation_spaces[agent_id]
act_space = env.action_spaces[agent_id]

config = (
    PPOConfig()
    .reporting(
        metrics_num_episodes_for_smoothing=10000
    )
    .environment(
        env='env',
        env_config=env_cfg,
    )
    .multi_agent(
      policies=[module_id],
      policy_mapping_fn=(lambda agent_id, episode, **kwargs: module_id),
      policies_to_train=[module_id],
    )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                module_id: RLModuleSpec(
                    catalog_class=AttentionPPOCatalog,
                    observation_space=obs_space,
                    action_space=act_space,
                    model_config={
                        "attention_emb_dim": 128, #tune.choice([32, 64, 128]),
                        "attn_ff_dim": 2048, #tune.choice([512, 1024, 2048]),
                        "head_fcnet_hiddens": tuple([256,256]),
                        "vf_share_layers": False, #tune.choice([True, False]),
                        "full_transformer": False,
                        "attn_layers": 1,
                        "recursive": False,
                    },
                )
            }
        )
    )
    .training(
        minibatch_size=tune.choice([512, 1024, 2048, 4096]),
        train_batch_size=32768, #tune.sample_from(
        #    lambda spec: spec.config["minibatch_size"] * args.num_env_runners
        #),
        
        lr=tune.loguniform(1e-5, 1e-3),
        gamma=tune.uniform(0.95, 0.99),
        lambda_=tune.uniform(0.97, 1.0),
        clip_param=tune.uniform(0.1, 0.3),
        grad_clip=tune.choice([None, 40, 100, 200]),
        
        kl_target=tune.uniform(0.01, 0.03),
        kl_coeff=tune.uniform(0.1, 0.4),
        use_kl_loss=tune.choice([True, False]),
        
        vf_clip_param=tune.choice([10.0, 40.0, float("inf")]),
        vf_loss_coeff=tune.uniform(0.01, 1.0),
        vf_share_layers=False,
        
        learner_class=BatchedCriticPPOLearner,
        learner_config_dict={'critic_batch_size': 32768},
    )
    .learners(
        num_learners=0,
        num_gpus_per_learner=1.0/args.num_concurrent,
    )
    .env_runners(
        num_env_runners=args.num_env_runners,
    )
)

# Create Custom Progress Reporter
reporter = CustomReporter(
    metric_columns={
        'time_total_s': 'Seconds',
        "env_runners/episode_return_mean": 'Reward',
        'training_iteration': 'Iters',
    },
    max_report_frequency=10,
    metric='Reward', # The metric to sort tables by
    mode='max',
    time_col='Seconds',
    rounding={
        'Seconds': 0,
        'Reward': 3,
    }
)

# Create Checkpoint Config
config_checkpoint = train.CheckpointConfig(
    checkpoint_at_end=True,
    num_to_keep=10,
    checkpoint_frequency=20,
    checkpoint_score_order='max',
    checkpoint_score_attribute="env_runners/episode_return_mean",
)

# Create Tuner Config
config_tuner = tune.TuneConfig(
    metric="env_runners/episode_return_mean", # Metric to optimize.
    mode='max',
    trial_dirname_creator=lambda trial: f'{trial.trial_id}',
    search_alg=ConcurrencyLimiter(
        searcher=HyperOptSearch(),
        max_concurrent=args.num_concurrent,
    ),
    scheduler=ASHAScheduler(
        # Removes lowest 75% (by default) of performers at a number of cutoffs generated based on 
        # min and max time. At each time cutoff, reward is recorded.
        # 5, 20, 80 if max rungs = 3, min_t = 5, max_t = 300, rf = 4
        time_attr="training_iteration", # This is the issue! time_attr isn't showing up in results.
        grace_period=grace_period_iter, # Don't stop trials before this point
        max_t=max_iter_individual,      # Stop trials after this point, no matter what
        reduction_factor=2,             # Reduce by this factor at each milestone (a function of min/max t)
    ),
    num_samples=-1,
    time_budget_s=max_time_total,       # Timeout for all trials
)

# Create Tuner Object
os.environ['RAY_AIR_NEW_OUTPUT'] = '0'
tuner = tune.Tuner(
    "PPO",
    param_space=config,
    run_config=train.RunConfig(
        name=experiment_name,
        stop={ # Get us to the threshold (decent performance)
            "env_runners/episode_return_mean": args.stop_threshold,
        },
        storage_path=str(PROJECT_PATH / 'hyperparameter_opt/results'),
        checkpoint_config=config_checkpoint,
        progress_reporter=reporter,
        verbose=1,
    ),
    tune_config=config_tuner
)

# Start Training
tuner.fit()