import os
import copy
from collections import defaultdict
import typing
import torch
from pathlib import Path

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.core import (
    COMPONENT_LEARNER,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_RL_MODULE,
)
from ray.rllib.utils.test_utils import (
    check,
)

class LoadOnAlgoInitCallback(RLlibCallback):
    def __init__(
        self,
        ckpt_path: str,
        module_name: str,
        vf_cold_start: bool = False,
        dest_module_names: typing.Optional[typing.List[str]] = None,
        iters_to_warmup_new: int = -1,
    ):
        dest_module_names = dest_module_names or [module_name] # Potentially, load weights into new dest.
        self.ckpt_path = (
            Path(ckpt_path)  # <- algorithm's checkpoint dir
            / COMPONENT_LEARNER_GROUP  # <- learner group
            / COMPONENT_LEARNER  # <- learner
            / COMPONENT_RL_MODULE  # <- MultiRLModule
            / module_name  # <- (single) RLModule
        )
        self.dest_module_names = dest_module_names
        self.vf_cold_start = vf_cold_start # Don't load the critic's parameters
        self.iters_to_warmup_new = iters_to_warmup_new
        
    def on_algorithm_init(self, *, algorithm, **kwargs):
        print('='*10)
        print(f"Loading checkpoint: {self.ckpt_path}")
        print('='*10)
        original_states = {}
        new_parameters = defaultdict(lambda: []) # module_name -> [parameter_names]
        for dest_module in self.dest_module_names:
            m_to_load = algorithm.learner_group._learner._module[dest_module]
            original_state = copy.deepcopy(m_to_load.state_dict())
            if (self.vf_cold_start):
                print("COLD STARTING VF")
                keys_to_load = filter(lambda x: x[:2]=='vf' or x[8:14]=='critic', list(original_state.keys()))
                original_states[dest_module] = {k: original_state[k] for k in keys_to_load}
            if (self.iters_to_warmup_new != -1):
                print(f"================================ ORIGINAL STATE: {original_state['encoder.critic_encoder.mha.0.ff.3.bias']}")
            # Cold start: Could edit restore_from_path for cold start; remember that Algorithm has its own override.
            # Could also create a temp file with the vf weights stripped out, like in https://github.com/ray-project/ray/blob/master/rllib/algorithms/tests/test_algorithm_rl_module_restore.py
            algorithm.restore_from_path(
                self.ckpt_path,
                component=(
                    COMPONENT_LEARNER_GROUP
                    + "/"
                    + COMPONENT_LEARNER
                    + "/"
                    + COMPONENT_RL_MODULE
                    + f"/{dest_module}"
                ),
            )
            new_state = m_to_load.state_dict()
            if (self.iters_to_warmup_new != -1):
                print(f"================================ NEW STATE: {new_state['encoder.critic_encoder.mha.0.ff.3.bias']}")
            # Build a list of parameters that *weren't* changed. Those are our new values.
            if (self.iters_to_warmup_new != -1):
                for (on, op), (nn, np) in zip(original_state.items(), new_state.items()):
                    if (torch.equal(op, np)):
                        print(f'{on} unchanged after load. Marked as new.')
                        new_parameters[dest_module].append(nn)
        # Now, set weight on VF to initial values if we're cold-starting them.
        if (self.vf_cold_start):
            print("Load original state")
            def reload_original_vf(_learner):
                for dest_module in self.dest_module_names:
                 _learner.module[dest_module].load_state_dict(original_states[dest_module], strict=False)
            algorithm.learner_group.foreach_learner(func=reload_original_vf)
        # If we're importing weights that aren't a perfect match, note the discrepancy and deactivate the weights we loaded for K epochs
        if (self.iters_to_warmup_new != -1):
            print(f'Novel parameters: {new_parameters}')
            def freeze_trained_parameters(_learner):
                for dest_module in self.dest_module_names:
                    module = _learner.module[dest_module]
                    param_dict = {name: param for name, param in module.named_parameters()}
                    if (self.iters_to_warmup_new > 0):
                        for (name, param) in param_dict.items():
                            if (name not in new_parameters[dest_module]): # Freeze the parameters that were loaded.
                                print(f'Freezing parameter: {name}')
                                param.requires_grad=False
                            else:
                                print(f'NOT Freezing parameter: {name}')
            algorithm.learner_group.foreach_learner(func=freeze_trained_parameters)
    
    def on_train_result(self,*,
        algorithm: "Algorithm",
        metrics_logger: typing.Optional[MetricsLogger] = None,
        result: dict,
        **kwargs,
    ) -> None:
        if (self.iters_to_warmup_new==-1):
            return
        print(f'on_train_result {algorithm.iteration}')
        ### Reactivate frozen parameters when ready
        if (algorithm.iteration >= self.iters_to_warmup_new):
          def unfreeze_parameters(_learner):
            print("Unfreezing parameters")
            for dest_module in self.dest_module_names:
              module = _learner.module[dest_module]
              param_dict = {name: param for name, param in module.named_parameters()}
              for (name, param) in param_dict.items():
                if (param.requires_grad==False):
                  print(f'Reactivating {name}')
                param.requires_grad=True
          algorithm.learner_group.foreach_learner(func=unfreeze_parameters)
          self.iters_to_warmup_new = -1