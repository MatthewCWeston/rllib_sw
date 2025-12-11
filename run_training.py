import sys
import importlib.util
import json
import torch
import functools
import numpy as np

import ray
from ray.rllib.models import ModelCatalog

from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.connectors.env_to_module.flatten_observations import FlattenObservations
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EPISODE_RETURN_MIN,
    TRAINING_ITERATION_TIMER,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)

from classes.attention_encoder import AttentionPPOCatalog
from classes.run_tune_training import run_tune_training
from classes.curiosity import add_curiosity
from classes.batched_critic_ppo import BatchedCriticPPOLearner
from callbacks.checkpoint_restore_callback import assignRestoreCallback
from callbacks.curriculum_learning_callback import CurriculumLearningCallback
from callbacks.render_callback import RenderCallback


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
parser.add_argument("--use-lstm", action='store_true') # Use an LSTM (default arch only for now)
parser.add_argument("--curiosity", action='store_true') # Use intrinsic motivation
parser.add_argument("--curriculum", nargs="*",  # Use curriculum learning
    choices=["grav_multiplier", "size_multiplier", "target_speed", "target_ammo"], default=[])
parser.add_argument("--curriculum-increments", type=int, default=10) # Rate to phase in gravity
parser.add_argument("--curriculum-patience", type=int, default=10) # Episodes over threshold b4 promotion
parser.add_argument("--curriculum-score-threshold", type=float, default=1.1) # Threshold to promote size
parser.add_argument("--curriculum-allow-demotions", action='store_true') # Threshold to promote size
parser.add_argument("--share-layers", action='store_true') # Only applies to custom architecture
parser.add_argument("--lr", type=float, default=1e-6) 
parser.add_argument("--lr-half-life", type=float) # Epochs for LR to halve, for exponential decay
parser.add_argument("--vf-clip", type=str, default='40.0')
parser.add_argument("--gamma", type=float, default=.999) # Reward discount over time
parser.add_argument("--lambda_", type=float, default=0.8) # Bootstrapping ratio (lower=more bootstrapped)
parser.add_argument("--attn-dim", type=int, default=128) # Encoder dimensionality
parser.add_argument("--attn-ff-dim", type=int, default=2048) # Feedforward component of attention layers
parser.add_argument("--attn-layers", type=int, default=1) # Times to recursively run our attention layer
parser.add_argument("--full-transformer", action='store_true') # Use full Transformer layers from PyTorch
parser.add_argument("--attn-recursive", action='store_true')
parser.add_argument('--fcnet', nargs='+', type=int, default=[256,256]) # Head architecture
parser.add_argument("--batch-size", type=int, default=32768)
parser.add_argument("--minibatch-size", type=int, default=4096)
parser.add_argument("--critic-batch-size", type=int, default=32768)
parser.add_argument("--render-every", type=int, default=0) # Every X steps, record a video
parser.add_argument("--restore-checkpoint", type=str)

args = parser.parse_args()

env_name = args.env_name
target_env = get_env_class(env_name)
register_env("env", lambda cfg: target_env(cfg))

# Configure run
callbacks = []
config = (
    PPOConfig()
    .environment(
        env="env",
        env_config=args.env_config,
    )
    .framework("torch")
    .training(
        train_batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        lr=args.lr,
        vf_clip_param=float(args.vf_clip),
        use_kl_loss=False,  # From hyperparameter search
        grad_clip=100,      # From hyperparameter search
        lambda_=args.lambda_,
        learner_class=BatchedCriticPPOLearner,
        learner_config_dict={'critic_batch_size': args.critic_batch_size}, # Just to avoid OOM; not a hyperparameter
    )
)
# Handle envs in MA format
env = target_env(args.env_config) # sample
ma_env = isinstance(env, MultiAgentEnv)
if (ma_env):
    agent_id = env.agents[0]
    module_id = 'my_policy'
    obs_space = env.observation_spaces[agent_id]
    act_space = env.action_spaces[agent_id]
    config.multi_agent(
      policies=[module_id],
      policy_mapping_fn=(lambda agent_id, episode, **kwargs: module_id),
      policies_to_train=[module_id],
    )
else:
    module_id = DEFAULT_MODULE_ID
    obs_space = env.observation_space
    act_space = env.action_space

# Architecture
if (not args.no_custom_arch):
    print('Using custom architecture')
    print(f"Share layers = {args.share_layers}")
    specs = {
        module_id: RLModuleSpec(
            catalog_class=AttentionPPOCatalog,
            observation_space=obs_space,
            action_space=act_space,
            model_config={
                "attention_emb_dim": args.attn_dim,
                "full_transformer": args.full_transformer,
                "attn_layers": args.attn_layers,
                "attn_ff_dim": args.attn_ff_dim,
                "recursive": args.attn_recursive,
                "head_fcnet_hiddens": tuple(args.fcnet),
                "vf_share_layers": args.share_layers,
            },
        )
    }
    config.env_runners(
        num_env_runners=args.num_env_runners,
    )
else:
    print('Using default architecture')
    specs = {
        module_id: RLModuleSpec(
            model_config=DefaultModelConfig(
                use_lstm=args.use_lstm,
                lstm_cell_size=1024,
            ),
        )
    }
    config.env_runners(
        num_env_runners=args.num_env_runners,
        env_to_module_connector=(
            lambda env, spaces, device: FlattenObservations(multi_agent=ma_env)
        ),
    )
# Curiosity
if (args.curiosity):
    print('Using curiosity')
    add_curiosity(config, specs)
    
# Curriculum Learning
if (len(args.curriculum)>0):
    print('Using curriculum learning')
    callbacks.append(
        functools.partial(
            CurriculumLearningCallback,
            attributes=args.curriculum,
            env_config=args.env_config,
            promotion_thresholds={
                # Reduce target size when above a certain score
                "size_multiplier": (EPISODE_RETURN_MEAN, args.curriculum_score_threshold), 
                # Increase gravity when we reliably aren't crashing before achieving anything
                "grav_multiplier": (EPISODE_RETURN_MIN, 0.0),
                # Reduce target speed when above a certain score
                "target_speed": (EPISODE_RETURN_MEAN, args.curriculum_score_threshold),
                # Increase target ammo when above a certain score
                "target_ammo": (EPISODE_RETURN_MEAN, args.curriculum_score_threshold),
            },
            promotion_patience = args.curriculum_patience,
            num_increments = args.curriculum_increments,
            allow_demotions = args.curriculum_allow_demotions,
        )
    )

# Learning rate decay
if (args.lr_half_life): # Default to exponential lr decay
    lr_factor = np.e ** (np.log(0.5)/args.lr_half_life)
    print(f"lr factor (exponential) = {lr_factor}")
    config.experimental(
        # Add two learning rate schedulers to be applied in sequence.
        _torch_lr_scheduler_classes=[
            functools.partial(
                torch.optim.lr_scheduler.ExponentialLR, gamma=lr_factor
            )
        ]
    )
    
# Creating recordings
if (args.render_every > 0):
    callbacks.append(functools.partial(
            RenderCallback,
            render_every=args.render_every
        ))

# Add spec
config.rl_module(
    rl_module_spec=MultiRLModuleSpec(
        rl_module_specs=specs
    ),
)
# Add callbacks
config.callbacks(callbacks)

# Set the stopping arguments.
EPISODE_RETURN_MEAN_KEY = f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}"
stop = {
    TRAINING_ITERATION_TIMER: args.stop_iters,
    EPISODE_RETURN_MEAN_KEY: args.stop_reward,
}

# Load policy if applicable
if (args.restore_checkpoint):
    print(f"Restoring checkpoint: {args.restore_checkpoint}")
    assignRestoreCallback(args.restore_checkpoint, config, module_id)

''' 
algo = config.build_algo()

from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
)
import numpy as np

num_iters = args.stop_iters

for i in range(num_iters):
  results = algo.train()
  if ENV_RUNNER_RESULTS in results:
      mean_return = results[ENV_RUNNER_RESULTS].get(
          EPISODE_RETURN_MEAN, np.nan
      )
      print(f"iter={i} R={mean_return}") 
'''

# Run the experiment.
run_tune_training(config,args,stop=stop) #'''