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

def assignRestoreCallback(ckpt_path, config, module_name):
    # Convert path to absolute path to resolve PyArrow being finicky? You'll get "URI has empty scheme" if you try to use relative paths here.
    ckpt_path = os.path.abspath(ckpt_path)
    # 
    ckpt_path = (
            Path(ckpt_path)  # <- algorithm's checkpoint dir
            / COMPONENT_LEARNER_GROUP  # <- learner group
            / COMPONENT_LEARNER  # <- learner
            / COMPONENT_RL_MODULE  # <- MultiRLModule
            / module_name  # <- (single) RLModule
        )
    class LoadOnAlgoInitCallback(RLlibCallback):
        def on_algorithm_init(self, *, algorithm, **kwargs):
            print('='*10)
            print(f"Loading checkpoint: {ckpt_path}")
            print('='*10)
            m_to_load = algorithm.get_module(module_name)
            weight_before = convert_to_numpy(next(iter(m_to_load.parameters())))
            algorithm.restore_from_path(
                ckpt_path,
                component=(
                    COMPONENT_LEARNER_GROUP
                    + "/"
                    + COMPONENT_LEARNER
                    + "/"
                    + COMPONENT_RL_MODULE
                    + f"/{module_name}"
                ),
            )
            # Make sure weights were updated.
            weight_after = convert_to_numpy(next(iter(m_to_load.parameters())))
            check(weight_before, weight_after, false=True)
    config.callbacks(LoadOnAlgoInitCallback)