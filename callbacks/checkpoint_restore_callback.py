import os
import copy
import typing
from pathlib import Path

from ray.rllib.callbacks.callbacks import RLlibCallback
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
        vf_cold_start: bool,
        dest_module_names: typing.List[str] | None = None,
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
        
    def on_algorithm_init(self, *, algorithm, **kwargs):
        print('='*10)
        print(f"Loading checkpoint: {self.ckpt_path}")
        print('='*10)
        original_states = {}
        for dest_module in self.dest_module_names:
            m_to_load = algorithm.get_module(dest_module)
            if (self.vf_cold_start):
                original_states = copy.deepcopy(m_to_load.state_dict())
                keys_to_load = filter(lambda x: x[:2]=='vf' or x[8:14]=='critic', list(original_state.keys()))
                original_states[dest_module] = {k: original_state[k] for k in keys_to_load}
            weight_before = convert_to_numpy(next(iter(m_to_load.parameters())))
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
            # Make sure weights were updated.
            weight_after = convert_to_numpy(next(iter(m_to_load.parameters())))
            check(weight_before, weight_after, false=True)
        # Now, set weight on VF to initial values if we're cold-starting them.
        if (self.vf_cold_start):
            print("Load original state")
            def reload_original_vf(_learner):
                for dest_module in self.dest_module_names
                 _learner.module[dest_module].load_state_dict(original_states[dest_module], strict=False)
            algorithm.learner_group.foreach_learner(func=reload_original_vf)