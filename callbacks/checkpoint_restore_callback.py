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
            print(list(original_state.keys()))
            #keys_to_load = filter(lambda x: x[:2]=='vf', list(original_state.keys()))
            keys_to_load = filter(lambda x: x[:2]=='vf' or x[8:14]=='critic', list(original_state.keys()))
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
            
# ['encoder.actor_encoder.mha.0.mha.in_proj_weight', 'encoder.actor_encoder.mha.0.mha.in_proj_bias', 'encoder.actor_encoder.mha.0.mha.out_proj.weight', 'encoder.actor_encoder.mha.0.mha.out_proj.bias', 'encoder.actor_encoder.mha.0.norm_attn.weight', 'encoder.actor_encoder.mha.0.norm_attn.bias', 'encoder.actor_encoder.mha.0.norm_ff.weight', 'encoder.actor_encoder.mha.0.norm_ff.bias', 'encoder.actor_encoder.mha.0.residual.0.weight', 'encoder.actor_encoder.mha.0.residual.0.bias', 'encoder.actor_encoder.mha.0.residual.3.weight', 'encoder.actor_encoder.mha.0.residual.3.bias', 'encoder.actor_encoder.embs.missiles_friendly.weight', 'encoder.actor_encoder.embs.missiles_friendly.bias', 'encoder.actor_encoder.embs.missiles_hostile.weight', 'encoder.actor_encoder.embs.missiles_hostile.bias', 'encoder.actor_encoder.embs.opponent.weight', 'encoder.actor_encoder.embs.opponent.bias', 'encoder.actor_encoder.embs.self.weight', 'encoder.actor_encoder.embs.self.bias', 'encoder.critic_encoder.mha.0.mha.in_proj_weight', 'encoder.critic_encoder.mha.0.mha.in_proj_bias', 'encoder.critic_encoder.mha.0.mha.out_proj.weight', 'encoder.critic_encoder.mha.0.mha.out_proj.bias', 'encoder.critic_encoder.mha.0.norm_attn.weight', 'encoder.critic_encoder.mha.0.norm_attn.bias', 'encoder.critic_encoder.mha.0.norm_ff.weight', 'encoder.critic_encoder.mha.0.norm_ff.bias', 'encoder.critic_encoder.mha.0.residual.0.weight', 'encoder.critic_encoder.mha.0.residual.0.bias', 'encoder.critic_encoder.mha.0.residual.3.weight', 'encoder.critic_encoder.mha.0.residual.3.bias', 'encoder.critic_encoder.embs.missiles_friendly.weight', 'encoder.critic_encoder.embs.missiles_friendly.bias', 'encoder.critic_encoder.embs.missiles_hostile.weight', 'encoder.critic_encoder.embs.missiles_hostile.bias', 'encoder.critic_encoder.embs.opponent.weight', 'encoder.critic_encoder.embs.opponent.bias', 'encoder.critic_encoder.embs.self.weight', 'encoder.critic_encoder.embs.self.bias', 'pi.log_std_clip_param_const', 'pi.net.mlp.0.weight', 'pi.net.mlp.0.bias', 'pi.net.mlp.2.weight', 'pi.net.mlp.2.bias', 'pi.net.mlp.4.weight', 'pi.net.mlp.4.bias', 'vf.log_std_clip_param_const', 'vf.net.mlp.0.weight', 'vf.net.mlp.0.bias', 'vf.net.mlp.2.weight', 'vf.net.mlp.2.bias', 'vf.net.mlp.4.weight', 'vf.net.mlp.4.bias']