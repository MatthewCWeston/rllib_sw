import sys
import importlib.util
import json

import ray
from ray.rllib.models import ModelCatalog

import gymnasium as gym

from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)

from classes.repeated_wrapper import ObsVectorizationWrapper
from classes.attention_encoder import AttentionEncoderConfig

from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    TRAINING_ITERATION_TIMER,
)

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
parser.add_argument("--env-config", type=str, default='{}')
parser.add_argument("--env-name", type=str)

args = parser.parse_args()

env_name = args.env_name
target_env = get_env_class(env_name)
register_env("env", lambda cfg: ObsVectorizationWrapper(target_env(cfg)))

env_config = json.loads(args.env_config)

# Configure run

config = (
    PPOConfig()
    .environment(
        env="env",
        env_config=env_config,
    )
    .framework("torch")
    .training(
        train_batch_size=32768,
        minibatch_size=4096,
        gamma=0.99,
        lr=1e-5
    )
    .rl_module(
        rl_module_spec=RLModuleSpec(
            catalog_class=AttentionPPOCatalog,
            model_config={
                "attention_emb_dim": 128,
                "head_fcnet_hiddens": (256, 256),
                "vf_share_layers": False,
            },
        ),
    )
)
# Set the stopping arguments.
EPISODE_RETURN_MEAN_KEY = f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}"
stop = {
    TRAINING_ITERATION_TIMER: args.stop_iters,
    EPISODE_RETURN_MEAN_KEY: args.stop_reward,
}

# Run the experiment.
run_rllib_example_script_experiment(
    config,
    args,
    stop=stop,
    success_metric={
        f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
    },
)