import os
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
    ):
        #ckpt_path = os.path.abspath(ckpt_path)
        self.ckpt_path = (
                Path(ckpt_path)  # <- algorithm's checkpoint dir
                / COMPONENT_LEARNER_GROUP  # <- learner group
                / COMPONENT_LEARNER  # <- learner
                / COMPONENT_RL_MODULE  # <- MultiRLModule
                / module_name  # <- (single) RLModule
            )
        self.module_name = module_name
        
    def on_algorithm_init(self, *, algorithm, **kwargs):
        print('='*10)
        print(f"Loading checkpoint: {self.ckpt_path}")
        print('='*10)
        m_to_load = algorithm.get_module(self.module_name)
        weight_before = convert_to_numpy(next(iter(m_to_load.parameters())))
        algorithm.restore_from_path(
            self.ckpt_path,
            component=(
                COMPONENT_LEARNER_GROUP
                + "/"
                + COMPONENT_LEARNER
                + "/"
                + COMPONENT_RL_MODULE
                + f"/{self.module_name}"
            ),
        )
        # Make sure weights were updated.
        weight_after = convert_to_numpy(next(iter(m_to_load.parameters())))
        check(weight_before, weight_after, false=True)