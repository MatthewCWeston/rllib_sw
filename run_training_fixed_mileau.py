'''
	A test script to train a single agent against a fixed random mileau of policies.
'''
import os
import sys
import importlib.util
import json
import torch
import functools
import numpy as np
from datetime import datetime

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
from classes.batched_critic_ppo import BatchedCriticPPOLearner
from callbacks.pfsp_callback import MAIN_MODULE, MAX_OPPONENTS

from environments.SW_1v1_env import SW_1v1_env

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
# Hyperparameters
parser.add_argument("--batch-size", type=int, default=32768)
parser.add_argument("--minibatch-size", type=int, default=4096)
parser.add_argument("--critic-batch-size", type=int, default=32768)
parser.add_argument("--lr", type=float, default=1e-6) 
parser.add_argument("--vf-clip", type=str, default='40.0')
parser.add_argument("--grad-clip", type=float, default=100.0)
parser.add_argument("--gamma", type=float, default=.999) # Reward discount over time
parser.add_argument("--lambda_", type=float, default=0.8) # Bootstrapping ratio (lower=more bootstrapped)
# We'll need to load in a checkpoint, and it might be beneficial to cold-start the value function
parser.add_argument("--opponents-path", type=os.path.abspath) # Folder containing a directory of checkpoints
parser.add_argument("--vf-cold-start", type=int, default=0) # Don't restore value function weights
# Multiple agents?
parser.add_argument("--no-load-main", action="store_true") # When loading, don't restore the learning module.
parser.add_argument("--identity-aug", action="store_true") # Augment the critic with the opposing agent's identity
# Miscellaneous/logging
parser.add_argument("--render-every", type=int, default=0) # Every X steps, record a video
parser.add_argument("--elo-eval", action="store_true")
parser.add_argument("--experiment-name", type=str, default="SPACEWAR_multiplayer")
parser.add_argument("--trial-name", type=str, default=datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
parser.add_argument("--results-path", type=os.path.abspath, default="./results_tmp/")
# Resources
parser.add_argument("--gpus-per-learner", type=float, default=1.0) # Remainder will be given to env runners
parser.add_argument("--cpus-per-env-runner", type=float, default=1.0) # Remainder will be given to env runners
parser.add_argument("--envs-per-env-runner", type=int, default=4)
parser.add_argument("--remote-worker-envs", action='store_true')

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
	.training(
		train_batch_size=args.batch_size,
		minibatch_size=args.minibatch_size,
		gamma=args.gamma,
		lr=args.lr,
		vf_clip_param=float(args.vf_clip),
		use_kl_loss=False,	# From hyperparameter search
		lambda_=args.lambda_,
		learner_class=BatchedCriticPPOLearner,
		learner_config_dict={
			'critic_batch_size': args.critic_batch_size, # Just to avoid OOM; not a hyperparameter
			'vf_cold_start': args.vf_cold_start, # Pre-train the value function for K minibatches
		},
		grad_clip=args.grad_clip if hasattr(args, 'grad_clip') else None,
		grad_clip_by="global_norm",
	)
	.learners(
		num_gpus_per_learner=args.gpus_per_learner,
	)
	.env_runners(
		num_cpus_per_env_runner=args.cpus_per_env_runner,
		num_gpus_per_env_runner=0 if args.num_env_runners==0 else (torch.cuda.device_count() - args.gpus_per_learner) / args.num_env_runners,
		num_envs_per_env_runner=args.envs_per_env_runner,
		num_env_runners=args.num_env_runners,
		remote_worker_envs=args.remote_worker_envs,
	)
)
# Handle envs in MA format
env = target_env(args.env_config) # sample
agent_id = env.agents[0]
act_space = env.action_spaces[agent_id]

# Architecture
opponents = [f for f in os.listdir(args.opponents_path) if os.path.isdir(os.path.join(args.opponents_path, f))]
specs = {}
modules = [MAIN_MODULE] + opponents
def atm_fn(agent_id, episode, **kwargs): # Select a random opponent
	if (agent_id == 0):
		return MAIN_MODULE
	else:
		eid = abs(hash(episode.id_))
		return opponents[eid%len(opponents)]

lrelu_override = args.activation_fn=="leakyrelu"
for a in modules:
	specs[a] = RLModuleSpec(
		catalog_class=AttentionPPOCatalog,
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
	
config.multi_agent(
	policies=modules,
	policy_mapping_fn=atm_fn,
	policies_to_train=[MAIN_MODULE], # Only the learned policy should be trained.
)
		
# Load opponent policies (main is initialized from scratch)
from callbacks.checkpoint_restore_callback import LoadOnAlgoInitCallback
print(f"Loading opponents: {args.opponents_path}; {opponents}")
for o in opponents:
	dest_modules = []
	callbacks.append(functools.partial(
		LoadOnAlgoInitCallback,
		ckpt_path=os.path.join(args.opponents_path, o),
		module_name=MAIN_MODULE,
		dest_module_names=[o],
	))

if (args.identity_aug): # Add a unique identity value to the critic's observations.
	from callbacks.augment_critic_with_id import AugmentCriticWithOpponentID
	print("Adding identity augmentation")
	def module_name_to_id(mname):
		if ('_0' not in mname):
			return 0
		else:
			return int(mname.split('_0')[-1])+1
	config.env_runners(
		env_to_module_connector=lambda env, spaces, device: [
			AugmentCriticWithOpponentID(
				module_name_to_id=module_name_to_id,
				max_opponents=MAX_OPPONENTS,
				),
		],
	)

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
	  vf_loss = results['learners'][MAIN_MODULE]['vf_loss']
	  mean_return = [(k, f'{v:.2f}') for k, v in mean_return.items()]
	  print(f"iter={i+1} VF loss={vf_loss:.2f} R={mean_return}")
'''

# Run the experiment.
run_tune_training(config,args,stop={TRAINING_ITERATION_TIMER: args.stop_iters,}) #'''