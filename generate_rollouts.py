# Load a checkpoint and generate some number of rollouts. Store obs, action, reward, term, trunc
from classes.inference_helpers import load_checkpoint, query_model
from environments.SW_MultiShoot_Env import SW_MultiShoot_Env
from classes.repeated_wrapper import ObsVectorizationWrapper

import numpy as np
import torch

checkpoint = '/mnt/c/Users/USER/Documents/SPACEWAR/SPACEWAR_module/checkpoints/reward_shaping_wip'
cfg = {"speed": 2.0, "ep_length": 256}

BATCH_SIZE = 32768 * 2

# Instantiate agent
agent = load_checkpoint(checkpoint)
# Instantiate environment
env = SW_MultiShoot_Env(cfg)

'''# Run X number of total steps
done = True
data = {'obs':[],'a':[],'r':[],'term':[],'trunc':[]}
print("ROLLOUTS START")
while len(data['obs']) < BATCH_SIZE:
    if (done):
        o, _ = env.reset()
    data['obs'].append(ObsVectorizationWrapper.serialize_obs(o, env.observation_space))
    a = query_model(agent, o, env)
    data['a'].append([ai.item() for ai in a])
    o, r, term, trunc, _ = env.step(a)
    data['term'].append(term)
    data['trunc'].append(trunc)
    data['r'].append(r)
    done = term or trunc
print("ROLLOUTS DONE")

# Save pkl file of dictionary with obs, action, reward, term, trunc.
for k, v in data.items():
    if isinstance(v[0], np.ndarray):
        data[k] = np.stack(v)
    else:
        data[k] = np.array(v)
np.save('./tmp/rollout.npy', data)'''
torch.save(agent.state_dict(), './tmp/agent.pt')