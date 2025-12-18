import os
import copy
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
        self.vf_cold_start = vf_cold_start # Don't load the critic's parameters
        
    def on_algorithm_init(self, *, algorithm, **kwargs):
        print('='*10)
        print(f"Loading checkpoint: {self.ckpt_path}")
        print('='*10)
        m_to_load = algorithm.get_module(self.module_name)
        if (self.vf_cold_start):
            original_state = copy.deepcopy(m_to_load.state_dict())
            keys_to_load = filter(lambda x: x[:2]=='vf', list(original_state.keys()))
            #keys_to_load = list(original_state.keys())
            original_state = {k: original_state[k] for k in keys_to_load}
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
                + f"/{self.module_name}"
            ),
        )
        # Make sure weights were updated.
        weight_after = convert_to_numpy(next(iter(m_to_load.parameters())))
        check(weight_before, weight_after, false=True)
        # Now, set weight on VF to initial values if we're resetting them.
        if (self.vf_cold_start):
            print("Load original state")
            #
            algorithm.learner_group.foreach_learner(
                func=lambda _learner: _learner.module[self.module_name].load_state_dict(original_state, strict=False)
            )
            # temporary, for making sure the weights get propagated.
            '''from ray.rllib.utils.metrics import ALL_MODULES
            algorithm.env_runner_group.sync_weights(
                from_worker_or_learner_group=algorithm.learner_group,
                policies=[ALL_MODULES],
                inference_only=True,
            )'''