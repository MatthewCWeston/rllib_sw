'''
    Taking a path to a folder, we calculate the ELO of each checkpoint within that folder.\
    
    python calculate_elo.py --ckpt-path "./checkpoints_release"
    
    # So, our final SARL agent is clearly stronger than its predecessors, but by no means perfect:
    1v1_basic_solved_moving_kindofsolved: W/L/D: 22.0%/70.0%/08.0%
    1v1_basic_solved_single_agent_final: W/L/D: 18.0%/76.0%/06.0%
    moving_kindofsolved_single_agent_final: W/L/D: 18.0%/78.0%/04.0%
'''

import os
import sys
from PIL import ImageDraw
import importlib.util
import numpy as np
import json
import argparse
from collections import defaultdict
from tqdm import tqdm

from classes.inference_helpers import load_checkpoint, query_model, query_value
from environments.SW_1v1_env import SW_1v1_env

DEFAULT_ACTION = "default"
WIN, LOSS, DRAW = 0, 1, 2

def match_agents(a1, a2, env):
    ''' Returns WIN, LOSS, or DRAW, depending on whether a1 beat a2 '''
    total_rewards = defaultdict(lambda:0)
    done = False
    o, _ = env.reset()
    agents = [a1,a2]
    while done==False:
        a = {k: query_model(agents[k], o[k], env, env.action_spaces[k]) for k in o.keys()}
        o, r, term, trunc, _ = env.step(a)
        for k, v in r.items():
            total_rewards[k] += v
        done = term['__all__'] or trunc['__all__']
    if (total_rewards[0] > total_rewards[1]):
        return WIN
    elif (total_rewards[0] == total_rewards[1]):
        return DRAW
    return LOSS

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt-path", type=str, default=None, help="Path to checkpoint folder")
args = parser.parse_args() 

# Load each agent from checkpoints
ckpt_path = os.path.abspath(args.ckpt_path)
agents = [] # [(name, model), ...]
for folder in os.listdir(ckpt_path):
    load_path = os.path.join(ckpt_path, folder)
    if os.path.isdir(load_path):
        agents.append((folder, load_checkpoint(load_path, 'my_policy')))

env = SW_1v1_env({"speed": 5, "ep_length": 4096, "egocentric": True})

# temp: Just record rates. a1-a2 -> W/L/D -> count
match_results = defaultdict(lambda: np.array([0,0,0]))

for _ in tqdm(range(50)):
    for i, (name1, a1) in enumerate(agents):
        for _, (name2, a2) in enumerate(agents[i+1:]):
            res = match_agents(a1, a2, env)
            match_results[f'{name1}_{name2}'][res] += 1
            
for match, res in match_results.items():
    res = res / res.sum()
    print(f"{match}: W/L/D: {res[WIN]*100:04.1f}%/{res[LOSS]*100:04.1f}%/{res[DRAW]*100:04.1f}%")