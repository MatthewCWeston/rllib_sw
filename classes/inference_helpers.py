import os
import torch

from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.columns import Columns

from classes.repeated_wrapper import ObsVectorizationWrapper

def load_checkpoint(ckpt_path):
    ckpt = os.path.join(
        ckpt_path,
        "learner_group",
        "learner",
        "rl_module",
        DEFAULT_MODULE_ID,
    )
    return RLModule.from_checkpoint(ckpt)

def query_model(agent, obs, env):
    # Model expects a vector instead of a dictionary
    obs = ObsVectorizationWrapper.serialize_obs(obs, env.observation_space)
    input_dict = {Columns.OBS: torch.tensor(obs).unsqueeze(0)}
    #
    action_shape = tuple(env.action_space.nvec)
    module_out = agent.forward_inference(input_dict)
    a = module_out['action_dist_inputs'][0]
    p = 0
    actions = []
    for ax in action_shape:
        actions.append(a[p:p+ax].argmax())
        p += ax
    return actions
    
def query_value(agent, obs, env):
    obs = ObsVectorizationWrapper.serialize_obs(obs, env.observation_space)
    input_dict = {Columns.OBS: torch.tensor(obs).unsqueeze(0)}
    #
    module_out = agent.compute_values(input_dict)
    return module_out.item()