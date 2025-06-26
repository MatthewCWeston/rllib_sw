import sys
import importlib.util
import json

import ray
from ray.rllib.models import ModelCatalog

from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
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
from classes.attention_encoder import AttentionPPOCatalog
from classes.run_tune_training import run_tune_training
from classes.curiosity import add_curiosity



# Get environment class 
def get_env_class(env_name):
    spec = importlib.util.spec_from_file_location(env_name, f'./environments/{env_name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_class = getattr(module, env_name)
    return agent_class

# Handle arguments

parser = add_rllib_example_script_args(default_reward=40, default_iters=50)
parser.set_defaults(
    enable_new_api_stack=True,
    num_env_runners=0
)
parser.add_argument("--env-config", type=json.loads, default={})
parser.add_argument("--env-name", type=str)
parser.add_argument("--no-custom-arch", action='store_true') # Don't use the attention-based encoder.
parser.add_argument("--curiosity", action='store_true') # Use intrinsic motivation

args = parser.parse_args()

env_name = args.env_name
target_env = get_env_class(env_name)
register_env("env", lambda cfg: ObsVectorizationWrapper(target_env(cfg)))

# Configure run

config = (
    PPOConfig()
    .environment(
        env="env",
        env_config=args.env_config,
    )
    .env_runners(
        num_env_runners=args.num_env_runners
    )
    .framework("torch")
    .training(
        train_batch_size=32768,
        minibatch_size=4096,
        gamma=0.99,
        lr=1e-5,
        vf_clip_param=40.0
    )
)
# Architecture
if (not args.no_custom_arch):
    print('Using custom architecture')
    specs = {
        DEFAULT_MODULE_ID: RLModuleSpec(
            catalog_class=AttentionPPOCatalog,
            model_config={
                "attention_emb_dim": 128,
                "head_fcnet_hiddens": (256, 256),
                "vf_share_layers": False, # See if True works better
            },
        )
    }
else:
    print('Using default architecture')
    specs = {
        DEFAULT_MODULE_ID: RLModuleSpec(
            model_config=DefaultModelConfig()
        )
    }
# Curiosity
if (args.curiosity):
    print('Using curiosity')
    add_curiosity(config, specs)
# Add spec
config.rl_module(
    rl_module_spec=MultiRLModuleSpec(
        rl_module_specs=specs
    ),
)

# Set the stopping arguments.
EPISODE_RETURN_MEAN_KEY = f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}"
stop = {
    TRAINING_ITERATION_TIMER: args.stop_iters,
    EPISODE_RETURN_MEAN_KEY: args.stop_reward,
}

# Run the experiment.
run_tune_training(config,args,stop=stop)