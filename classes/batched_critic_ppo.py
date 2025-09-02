# @title BatchedCriticPPOLearner
import abc
from typing import Any, List, Dict
import torch

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

# @title BatchedGeneralAdvantageEstimation
import numpy as np
from ray.rllib.connectors.learner import (
    GeneralAdvantageEstimation,
)

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.common.numpy_to_tensor import NumpyToTensor
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.postprocessing.value_predictions import compute_value_targets
from ray.rllib.utils.postprocessing.zero_padding import (
    split_and_zero_pad_n_episodes,
    unpad_data_if_necessary,
)
from ray.rllib.utils.typing import EpisodeType

class BatchedCriticPPOLearner(PPOTorchLearner):
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
              BatchedGeneralAdvantageEstimation(
                  gamma=self.config.gamma, lambda_=self.config.lambda_, batch_size=self.config.learner_config_dict["critic_batch_size"]
              )
          )
          
def batch_dict(m_in, s, mb_size):
    ''' Take a dictionary, start position, and batch size, and return a batch. '''
    to_return = {}
    # debug
    for k,v in m_in.items():
      if (type(v) is dict):
        to_return[k] = batch_dict(v, s, mb_size)
      else:
        to_return[k] = m_in[k][s:s+mb_size]
    return to_return

class BatchedGeneralAdvantageEstimation(GeneralAdvantageEstimation):
    """Learner ConnectorV2 piece computing GAE advantages and value targets on episodes.

    This subclass uses the minibatch size specified in the learner config to batch the value computations, preventing OOM errors when working with complex modules and large batch sizes.
    """
    def __init__(self,input_observation_space=None,input_action_space=None,*,gamma,lambda_, batch_size,):
        super().__init__(input_observation_space, input_action_space, gamma=gamma, lambda_=lambda_)
        self.batch_size = batch_size

    @override(ConnectorV2)
    def __call__(
        self,
        *,
        rl_module: MultiRLModule,
        episodes: List[EpisodeType],
        batch: Dict[str, Any],
        **kwargs,
    ):
        with torch.no_grad(): # We're computing advantages (for actor optimization) and targets (goals for vf optimization). I don't think either one needs gradients.
            # Device to place all GAE result tensors (advantages and value targets) on.
            device = None

            # Extract all single-agent episodes.
            sa_episodes_list = list(
                self.single_agent_episode_iterator(episodes, agents_that_stepped_only=False)
            )
            # Perform the value nets' forward passes.
            # TODO (sven): We need to check here in the pipeline already, whether a module
            #  should even be updated or not (which we usually do after(!) the Learner
            #  pipeline). This is an open TODO to move this filter into a connector as well.
            #  For now, we'll just check, whether `mid` is in batch and skip if it isn't.
            def computeValuesForModule(mid, module):
              # For now, we implement our own minibatching; I'm not sure there's a clean 1:1 match with anything already there.
              if mid in batch and isinstance(module, ValueFunctionAPI):
                  values = []
                  m_in = batch[mid]
                  for s in range(0, len(m_in[Columns.REWARDS]), self.batch_size):
                    v_mb = batch_dict(m_in, s, self.batch_size)
                    values.append(module.compute_values(v_mb))
                  values = torch.hstack(values)
                  return values
              return None

            vf_preds = rl_module.foreach_module(
                func=lambda mid, module: computeValuesForModule(mid, module),
                return_dict=True,
            )
            # Loop through all modules and perform each one's GAE computation.
            for module_id, module_vf_preds in vf_preds.items():
                # Skip those outputs of RLModules that are not implementers of
                # `ValueFunctionAPI`.
                if module_vf_preds is None:
                    continue

                module = rl_module[module_id]
                device = module_vf_preds.device
                # Convert to numpy for the upcoming GAE computations.
                module_vf_preds = convert_to_numpy(module_vf_preds)

                # Collect (single-agent) episode lengths for this particular module.
                episode_lens = [
                    len(e) for e in sa_episodes_list if e.module_id in [None, module_id]
                ]

                # Remove all zero-padding again, if applicable, for the upcoming
                # GAE computations.
                module_vf_preds = unpad_data_if_necessary(episode_lens, module_vf_preds)
                # Compute value targets.
                module_value_targets = compute_value_targets(
                    values=module_vf_preds,
                    rewards=unpad_data_if_necessary(
                        episode_lens,
                        convert_to_numpy(batch[module_id][Columns.REWARDS]),
                    ),
                    terminateds=unpad_data_if_necessary(
                        episode_lens,
                        convert_to_numpy(batch[module_id][Columns.TERMINATEDS]),
                    ),
                    truncateds=unpad_data_if_necessary(
                        episode_lens,
                        convert_to_numpy(batch[module_id][Columns.TRUNCATEDS]),
                    ),
                    gamma=self.gamma,
                    lambda_=self.lambda_,
                )
                assert module_value_targets.shape[0] == sum(episode_lens)

                module_advantages = module_value_targets - module_vf_preds
                # Drop vf-preds, not needed in loss. Note that in the DefaultPPORLModule,
                # vf-preds are recomputed with each `forward_train` call anyway to compute
                # the vf loss.
                # Standardize advantages (used for more stable and better weighted
                # policy gradient computations).
                module_advantages = (module_advantages - module_advantages.mean()) / max(
                    1e-4, module_advantages.std()
                )

                # Zero-pad the new computations, if necessary.
                if module.is_stateful():
                    module_advantages = np.stack(
                        split_and_zero_pad_n_episodes(
                            module_advantages,
                            episode_lens=episode_lens,
                            max_seq_len=module.model_config["max_seq_len"],
                        ),
                        axis=0,
                    )
                    module_value_targets = np.stack(
                        split_and_zero_pad_n_episodes(
                            module_value_targets,
                            episode_lens=episode_lens,
                            max_seq_len=module.model_config["max_seq_len"],
                        ),
                        axis=0,
                    )
                batch[module_id][Postprocessing.ADVANTAGES] = module_advantages
                batch[module_id][Postprocessing.VALUE_TARGETS] = module_value_targets

            # Convert all GAE results to tensors.
            if self._numpy_to_tensor_connector is None:
                self._numpy_to_tensor_connector = NumpyToTensor(
                    as_learner_connector=True, device=device
                )
            tensor_results = self._numpy_to_tensor_connector(
                rl_module=rl_module,
                batch={
                    mid: {
                        Postprocessing.ADVANTAGES: module_batch[Postprocessing.ADVANTAGES],
                        Postprocessing.VALUE_TARGETS: (
                            module_batch[Postprocessing.VALUE_TARGETS]
                        ),
                    }
                    for mid, module_batch in batch.items()
                    if vf_preds[mid] is not None
                },
                episodes=episodes,
            )
            # Move converted tensors back to `batch`.
            for mid, module_batch in tensor_results.items():
                batch[mid].update(module_batch)

            return batch