
# @title NGPPOLearner
import abc
from typing import Any, Dict

from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_CURR_ENTROPY_COEFF_KEY,
    LEARNER_RESULTS_KL_KEY,
    PPOConfig,
)
from ray.rllib.connectors.learner import (
    AddOneTsToEpisodesAndTruncate,
)
from ray.rllib.core.learner.learner import Learner
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.utils.annotations import (
    override,
    OverrideToImplementCustomLogic_CallToSuperRecommended,
)
from ray.rllib.utils.lambda_defaultdict import LambdaDefaultDict
from ray.rllib.utils.metrics import (
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_MODULE_STEPS_TRAINED,
)
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.typing import ModuleID, TensorType

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.algorithms.ppo.ppo_learner import PPOLearner

# @title NGGeneralAdvantageEstimation
from typing import Any, List, Dict

from ray.rllib.connectors.learner import (
    GeneralAdvantageEstimation,
)
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import EpisodeType
import torch

class NGGeneralAdvantageEstimation(GeneralAdvantageEstimation):
    """Learner ConnectorV2 piece computing GAE advantages and value targets on episodes.

    This subclass turns off grad when computing advantages and targets.
    """
    @override(ConnectorV2)
    def __call__(self,*,rl_module: MultiRLModule,episodes: List[EpisodeType],batch: Dict[str, Any],**kwargs,):
        with torch.no_grad():
          return super().__call__(rl_module=rl_module ,episodes=episodes, batch=batch, **kwargs)

class NGPPOLearner(PPOTorchLearner):
  @override(Learner)
  def build(self) -> None:
      TorchLearner.build(self) # We don't want to use super() because we're creating an alternative version of PPOLearner's build method, and we don't want to do certain things twice.

      # Dict mapping module IDs to the respective entropy Scheduler instance.
      self.entropy_coeff_schedulers_per_module: Dict[
          ModuleID, Scheduler
      ] = LambdaDefaultDict(
          lambda module_id: Scheduler(
              fixed_value_or_schedule=(
                  self.config.get_config_for_module(module_id).entropy_coeff
              ),
              framework=self.framework,
              device=self._device,
          )
      )

      # Set up KL coefficient variables (per module).
      # Note that the KL coeff is not controlled by a Scheduler, but seeks
      # to stay close to a given kl_target value.
      self.curr_kl_coeffs_per_module: Dict[ModuleID, TensorType] = LambdaDefaultDict(
          lambda module_id: self._get_tensor_variable(
              self.config.get_config_for_module(module_id).kl_coeff
          )
      )

      # Extend all episodes by one artificial timestep to allow the value function net
      # to compute the bootstrap values (and add a mask to the batch to know, which
      # slots to mask out).
      if (
          self._learner_connector is not None
          and self.config.add_default_connectors_to_learner_pipeline
      ):
          # Before anything, add one ts to each episode (and record this in the loss
          # mask, so that the computations at this extra ts are not used to compute
          # the loss).
          self._learner_connector.prepend(AddOneTsToEpisodesAndTruncate())
          # At the end of the pipeline (when the batch is already completed), add the
          # GAE connector, which performs a vf forward pass, then computes the GAE
          # computations, and puts the results of this (advantages, value targets)
          # directly back in the batch. This is then the batch used for
          # `forward_train` and `compute_losses`.
          self._learner_connector.append(
              NGGeneralAdvantageEstimation(
                  gamma=self.config.gamma, lambda_=self.config.lambda_
              )
          )