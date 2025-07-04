import sys
import importlib.util
import json
import torch
import functools

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
from classes.batched_critic_ppo import BatchedCriticPPOLearner
from classes.checkpoint_restore_callback import assignRestoreCallback


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
    num_env_runners=0,
)
parser.add_argument("--env-config", type=json.loads, default={})
parser.add_argument("--env-name", type=str)
parser.add_argument("--no-custom-arch", action='store_true') # Don't use the attention-based encoder.
parser.add_argument("--curiosity", action='store_true') # Use intrinsic motivation
parser.add_argument("--share-layers", action='store_true') # Only applies to custom architecture
parser.add_argument("--lr", type=float, default=1e-6) 
parser.add_argument("--lr-final", type=float) # Reward at end of training, if we want to change it.
parser.add_argument("--vf-clip", type=str, default='inf')
parser.add_argument("--gamma", type=float, default=.999)
parser.add_argument("--attn-dim", type=int, default=16) # Encoder dimensionality
parser.add_argument('--fcnet', nargs='+', type=int, default=[256,256]) # Head architecture
parser.add_argument("--batch-size", type=int, default=32768)
parser.add_argument("--minibatch-size", type=int, default=4096)
parser.add_argument("--critic-batch-size", type=int, default=32768)
parser.add_argument("--restore-checkpoint", type=str)

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
        num_env_runners=args.num_env_runners,
    )
    .framework("torch")
    .training(
        train_batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        lr=args.lr,
        vf_clip_param=float(args.vf_clip),
        learner_class=BatchedCriticPPOLearner,
        learner_config_dict={'critic_batch_size': args.critic_batch_size}, # Pass batch size here
    )
)
# Architecture
if (not args.no_custom_arch):
    print('Using custom architecture')
    print(f"Share layers = {args.share_layers}")
    specs = {
        DEFAULT_MODULE_ID: RLModuleSpec(
            catalog_class=AttentionPPOCatalog,
            model_config={
                "attention_emb_dim": args.attn_dim,
                "head_fcnet_hiddens": tuple(args.fcnet),
                "vf_share_layers": args.share_layers,
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
if (args.lr_final and (args.lr_final != args.lr)):
    lr_factor = (args.lr_final/args.lr)**(1/args.stop_iters) # divide by lr_final_scale over all epochs.
    print(f"lr factor = {lr_factor}")
    config.experimental(
        # Add two learning rate schedulers to be applied in sequence.
        _torch_lr_scheduler_classes=[
            functools.partial(
                torch.optim.lr_scheduler.ConstantLR,
                factor=lr_factor,
                total_iters=args.stop_iters,
            )
        ]
    )
    
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

# Load policy if applicable
if (args.restore_checkpoint):
    print(f"Restoring checkpoint: {args.restore_checkpoint}")
    assignRestoreCallback(args.restore_checkpoint, config)

# Run the experiment.
run_tune_training(config,args,stop=stop)