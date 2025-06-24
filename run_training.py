import sys
import importlib.util
import json

import ray
from ray.rllib.models import ModelCatalog

import gymnasium as gym

from ray import tune
from ray.tune.schedulers.pb2 import PB2
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    TRAINING_ITERATION_TIMER,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)

from classes.repeated_wrapper import ObsVectorizationWrapper
from classes.attention_encoder import AttentionEncoderConfig
from classes.run_tune_training import run_tune_training



# Get environment class 
def get_env_class(env_name):
    spec = importlib.util.spec_from_file_location(env_name, f'./environments/{env_name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_class = getattr(module, env_name)
    return agent_class

class AttentionPPOCatalog(PPOCatalog):
    """
    A special PPO catalog producing an encoder that handles dictionaries of (potentially Repeated) action spaces in the same manner as https://arxiv.org/abs/1909.07528.
    """

    @classmethod
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        **kwargs,
    ):
        return AttentionEncoderConfig(observation_space, **kwargs)

# Handle arguments

parser = add_rllib_example_script_args(default_reward=40, default_iters=50)
parser.set_defaults(
    enable_new_api_stack=True
)
parser.add_argument("--env-config", type=json.loads, default={})
parser.add_argument("--env-name", type=str)
parser.add_argument("--no-custom-arch", action='store_true') # Don't use the attention-based encoder.

args = parser.parse_args()

env_name = args.env_name
target_env = get_env_class(env_name)
register_env("env", lambda cfg: ObsVectorizationWrapper(target_env(cfg)))

# Configure run

'''config = (
    PPOConfig()
    .environment(
        env="env",
        env_config=args.env_config,
    )
    .framework("torch")
    .training(
        train_batch_size=32768,
        minibatch_size=4096,
        gamma=0.99,
        lr=1e-5
        lambda_=0.95,
        clip_param=0.2,
    )
)'''

pb2_scheduler = PB2(
    time_attr=f"{NUM_ENV_STEPS_SAMPLED_LIFETIME}",
    metric="env_runners/episode_return_mean",
    mode="max",
    perturbation_interval=50000,
    # Copy bottom % with top % weights.
    quantile_fraction=0.25,
    hyperparam_bounds={
        "lr": [1e-5, 1e-3],
        "gamma": [0.95, 0.99],
        "lambda": [0.97, 1.0],
        "entropy_coeff": [0.0, 0.01],
        "vf_loss_coeff": [0.01, 1.0],
        "clip_param": [0.1, 0.3],
        "kl_target": [0.01, 0.03],
        "minibatch_size": [512, 4096],
        "num_epochs": [6, 32],
        "use_kl_loss": [False, True],
        "kl_coeff": [0.1, 0.4],
        "vf_clip_param": [10.0, float("inf")],
        "grad_clip": [40, 200],
    },
)

config = (
    PPOConfig()
    .environment(
        env="env",
        env_config=args.env_config,
    )
    .framework("torch")
    .training(
        lr=tune.uniform(1e-5, 1e-3),
        gamma=tune.uniform(0.95, 0.99),
        lambda_=tune.uniform(0.97, 1.0),
        entropy_coeff=tune.choice([0.0, 0.01]),
        vf_loss_coeff=tune.uniform(0.01, 1.0),
        clip_param=tune.uniform(0.1, 0.3),
        kl_target=tune.uniform(0.01, 0.03),
        minibatch_size=tune.choice([512, 1024, 2048, 4096]),
        num_epochs=tune.randint(200, 400), # Going to have to figure out what to do with this. I need a long training run.
        use_kl_loss=tune.choice([True, False]),
        kl_coeff=tune.uniform(0.1, 0.4),
        vf_clip_param=tune.choice([10.0, 40.0, float("inf")]),
        grad_clip=tune.choice([None, 40, 100, 200]),
        train_batch_size=tune.sample_from(
            lambda spec: spec.config["minibatch_size"] * (args.num_env_runners or 1)
        ),
    )
)
# 
if (not args.no_custom_arch):
    print('Using custom architecture')
    config.rl_module(
        rl_module_spec=RLModuleSpec(
            catalog_class=AttentionPPOCatalog,
            model_config={
                "attention_emb_dim": 128,
                "head_fcnet_hiddens": (256, 256),
                "vf_share_layers": False,
            },
        ),
    )
else:
    print('Using default architecture')
    config.rl_module(rl_module_spec=RLModuleSpec(
          model_config={
              "head_fcnet_hiddens": (256,256),
              'vf_share_layers': False,
          }
        ),
    )
# Set the stopping arguments.
EPISODE_RETURN_MEAN_KEY = f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}"
stop = {
    TRAINING_ITERATION_TIMER: args.stop_iters,
    EPISODE_RETURN_MEAN_KEY: args.stop_reward,
}

# Run the experiment.
run_tune_training(config,args,stop=stop, scheduler=pb2_scheduler)