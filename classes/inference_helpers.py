import os
import torch
from torch.distributions import Categorical

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.columns import Columns

def load_checkpoint(ckpt_path, module_id):
    ckpt = os.path.join(
        ckpt_path,
        "learner_group",
        "learner",
        "rl_module",
        module_id,
    )
    return RLModule.from_checkpoint(ckpt)

def query_model(agent, obs, env):
    # Model expects a vector instead of a dictionary
    input_dict = {Columns.OBS: torch.tensor(obs).unsqueeze(0)}
    #
    action_shape = tuple(env.action_space.nvec)
    module_out = agent.forward_inference(input_dict)
    a = module_out['action_dist_inputs'][0]
    p = 0
    actions = []
    for ax in action_shape:
        a_logits = a[p:p+ax]
        probs = torch.nn.Softmax()(a_logits)
        action = Categorical(probs).sample()
        actions.append(action.item())
        p += ax
    return actions
    
def query_value(agent, obs, env):
    input_dict = {Columns.OBS: torch.tensor(obs).unsqueeze(0)}
    #
    module_out = agent.compute_values(input_dict)
    return module_out.item()