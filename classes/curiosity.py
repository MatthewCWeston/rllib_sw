from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.examples.rl_modules.classes.intrinsic_curiosity_model_rlm import IntrinsicCuriosityModel
from ray.rllib.examples.learners.classes.intrinsic_curiosity_learners import (
    ICM_MODULE_ID,
    PPOTorchLearnerWithCuriosity
)

from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import override

from gymnasium.spaces import MultiDiscrete, Box, Discrete
from torch import nn
from typing import Any, Dict

def add_curiosity(config, specs):
    '''
        Adds the requisite information to the config to allow it to use intrinsic motivation to explore.
    '''
    config.training(
      learner_config_dict={
          "intrinsic_reward_coeff": 1000.0, #0.05, # Weight for curiosity term
          "forward_loss_weight": 0.2, # Share of curiosity term contributed by forward loss (vs inverse dynamics loss)
      },
      learner_class=PPOTorchLearnerWithCuriosity,
    )
    # Use a different learning rate for training the ICM.
    config.rl_module(
        algorithm_config_overrides_per_module={
            ICM_MODULE_ID: AlgorithmConfig.overrides(lr=0.0005)
        },
    )
    # Add IC module
    specs[ICM_MODULE_ID] = RLModuleSpec(
        module_class=ICMTest,
        catalog_class=PPOCatalog, # So we can look up the action distributions
        learner_only=True, # Only create the ICM on the Learner workers
        model_config={
            "feature_dim": 288,
            "feature_net_hiddens": (256, 256),
            "feature_net_activation": "relu",
            "inverse_net_hiddens": (256, 256),
            "inverse_net_activation": "relu",
            "forward_net_hiddens": (256, 256),
            "forward_net_activation": "relu",
        },
    )
    
def get_action_space_size(a_s):
  if (type(a_s) == Discrete):
    return a_s.n
  elif (type(a_s) == MultiDiscrete):
    return a_s.nvec.sum()
  elif (type(a_s) == Box):
    return a_s.shape[0] # Assume 1D Box spaces.
  raise Exception("Unsupported action space")

# Notes: algo.env_runner.module.get_inference_action_dist_cls() works for multi.
# algo.env_runner.module.catalog._get_dist_cls_from_action_space is why it works for the main module. 
# .action_dist_cls is set in rl_module.__init__
# Accordingly, we set catalog to PPOCatalog to make this module work.
class ICMTest(IntrinsicCuriosityModel):
    @override(TorchRLModule)
    def setup(self):
        # Get the ICM achitecture settings from the `model_config` attribute:
        cfg = self.model_config
        feature_dim = cfg.get("feature_dim", 288)
        # Build the feature model (encoder of observations to feature space).
        layers = []
        dense_layers = cfg.get("feature_net_hiddens", (256, 256))
        # `in_size` is the observation space (assume a simple Box(1D)).
        in_size = self.observation_space.shape[0]
        for out_size in dense_layers:
            layers.append(nn.Linear(in_size, out_size))
            if cfg.get("feature_net_activation") not in [None, "linear"]:
                layers.append(
                    get_activation_fn(cfg["feature_net_activation"], "torch")()
                )
            in_size = out_size
        # Last feature layer of n nodes (feature dimension).
        layers.append(nn.Linear(in_size, feature_dim))
        self._feature_net = nn.Sequential(*layers)
        # Build the inverse model (predicting the action between two observations).
        layers = []
        dense_layers = cfg.get("inverse_net_hiddens", (256,))
        # `in_size` is 2x the feature dim.
        in_size = feature_dim * 2
        for out_size in dense_layers:
            layers.append(nn.Linear(in_size, out_size))
            if cfg.get("inverse_net_activation") not in [None, "linear"]:
                layers.append(
                    get_activation_fn(cfg["inverse_net_activation"], "torch")()
                )
            in_size = out_size
        # Last feature layer of n nodes (action space).
        action_space_size = get_action_space_size(self.action_space)
        layers.append(nn.Linear(in_size, action_space_size))
        self._inverse_net = nn.Sequential(*layers)
        # Build the forward model (predicting the next observation from current one and
        # action).
        layers = []
        dense_layers = cfg.get("forward_net_hiddens", (256,))
        # `in_size` is the feature dim + action space (one-hot).
        in_size = feature_dim + action_space_size
        for out_size in dense_layers:
            layers.append(nn.Linear(in_size, out_size))
            if cfg.get("forward_net_activation") not in [None, "linear"]:
                layers.append(
                    get_activation_fn(cfg["forward_net_activation"], "torch")()
                )
            in_size = out_size
        # Last feature layer of n nodes (feature dimension).
        layers.append(nn.Linear(in_size, feature_dim))
        self._forward_net = nn.Sequential(*layers)
        print("SETUP DONE")