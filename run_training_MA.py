'''
    For now, just optimize an agent against another agent initialized from the same weights
'''

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
from callbacks.checkpoint_restore_callback import LoadOnAlgoInitCallback
from callbacks.curriculum_learning_callback import CurriculumLearningCallback
from callbacks.render_callback import RenderCallback

from environments.SW_1v1_env import SW_1v1_env

# Handle arguments
parser = add_rllib_example_script_args(default_reward=40, default_iters=50)
parser.set_defaults(
    enable_new_api_stack=True,
    num_env_runners=0,
)
# Architecture should match that of the base agent
parser.add_argument("--attn-dim", type=int, default=128) # Encoder dimensionality
parser.add_argument("--attn-ff-dim", type=int, default=2048) # Feedforward component of attention layers
parser.add_argument('--fcnet', nargs='+', type=int, default=[256,256]) # Head architecture
# Hyperparameters
parser.add_argument("--batch-size", type=int, default=32768)
parser.add_argument("--minibatch-size", type=int, default=4096)
parser.add_argument("--critic-batch-size", type=int, default=32768)
parser.add_argument("--lr", type=float, default=1e-6) 
parser.add_argument("--vf-clip", type=str, default='40.0')
parser.add_argument("--gamma", type=float, default=.999) # Reward discount over time
parser.add_argument("--lambda_", type=float, default=0.8) # Bootstrapping ratio (lower=more bootstrapped)
# We'll need to load in a checkpoint, and it might be beneficial to cold-start the value function
parser.add_argument("--restore-checkpoint", type=str)
parser.add_argument("--vf-cold-start", type=int, default=0) # Don't restore value function weights
# Miscellaneous
parser.add_argument("--render-every", type=int, default=0) # Every X steps, record a video

args = parser.parse_args()

target_env = SW_1v1_env
register_env("env", lambda cfg: target_env(cfg))

# Configure run
callbacks = []
config = (
    PPOConfig()
    .environment(
        env="env",
        env_config={'speed':5.0,'ep_length':4096,'egocentric':True,},
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
        learner_config_dict={
            'critic_batch_size': args.critic_batch_size, # Just to avoid OOM; not a hyperparameter
            'vf_cold_start': args.vf_cold_start, # Pre-train the value function for K minibatches
        },
    )
)
# Handle envs in MA format
env = target_env(args.env_config) # sample
agent_id = env.agents[0]
module_id = 'my_policy'
obs_space = env.observation_spaces[agent_id]
act_space = env.action_spaces[agent_id]
config.multi_agent(
  policies=[module_id],
  policy_mapping_fn=(lambda agent_id, episode, **kwargs: module_id),
  policies_to_train=[module_id],
)

# Architecture
print('Using custom architecture')
print(f"Share layers = {args.share_layers}")
specs = {}

def atm_fn(agent_id, episode, **kwargs):
    eid = hash(episode.id_)
    if (agent_id==eid%2):
        return "main"
    return "main_v0"

for a in ['main', 'main_v0']:
    specs[a] = RLModuleSpec(
        catalog_class=AttentionPPOCatalog,
        observation_space=obs_space,
        action_space=act_space,
        model_config={
            "attention_emb_dim": args.attn_dim,
            "attn_ff_dim": args.attn_ff_dim,
            "head_fcnet_hiddens": tuple(args.fcnet),
            "vf_share_layers": args.share_layers,
            "attn_layers": 1,
            "full_transformer": False,
            "recursive": False,
        },
    )
    
config.multi_agent(
    policies=['main','main_v0'],
    policy_mapping_fn=atm_fn,
    policies_to_train=['main'], # Only the learned policy should be trained.
)
    
config.env_runners(
    num_env_runners=args.num_env_runners,
)
        
# Load policy if applicable.
if (args.restore_checkpoint):
    print(f"Restoring checkpoint: {args.restore_checkpoint}")
    callbacks.append(functools.partial(
        LoadOnAlgoInitCallback,
        ckpt_path=args.restore_checkpoint,
        module_name=module_id,
        vf_cold_start=args.vf_cold_start,
    ))
    
# Record matches every K epochs
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
    
#''' Test it out with this train loop
algo = config.build_algo()
num_iters = args.stop_iters

for i in range(num_iters):
  results = algo.train()
  if ENV_RUNNER_RESULTS in results:
      mean_return = results[ENV_RUNNER_RESULTS]['agent_episode_returns_mean']
      vf_loss = results['learners']['main']['vf_loss']
      mean_return = [(k, f'{v:.2f}') for k, v in mean_return.items()]
      print(f"iter={i+1} VF loss={vf_loss:.2f} R={mean_return}")
'''

# Run the experiment.
#run_tune_training(config,args,stop={TRAINING_ITERATION_TIMER: args.stop_iters,}) #'''