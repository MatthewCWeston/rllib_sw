'''
    Launch an instance of the SPACEWAR RL environment, with a pygame wrapper.
        
    We take a path to an rllib env with:
     - a render() function that returns a PIL image.
     - a getKeyMap() function that maps key inputs to actions within the environment.
'''

import os
import sys
from PIL import ImageDraw
import pygame
import importlib.util
import numpy as np
import json
import argparse
from collections import defaultdict
from classes.inference_helpers import load_checkpoint, query_model, query_value

DEFAULT_ACTION = "default"

env_name = sys.argv[1]

parser = argparse.ArgumentParser()
parser.add_argument("env-name", type=str, help="Name of the environment")
parser.add_argument("--env-config", type=json.loads, default={})
parser.add_argument("--ckpt-path", type=str, default=None, help="Path to checkpoint file, if applicable")
parser.add_argument("--ckpt-name", type=str, default="main")
args = parser.parse_args()

cfg = args.env_config
agent = None
if (args.ckpt_path is not None):
    ckpt_path = os.path.abspath(args.ckpt_path)
    agent = load_checkpoint(ckpt_path, args.ckpt_name)

def instantiate_env(env_name, cfg):
    spec = importlib.util.spec_from_file_location(env_name, f'./environments/{env_name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_class = getattr(module, env_name)
    return agent_class(cfg)
env = instantiate_env(env_name, cfg)
o, _ = env.reset()
size = env.size
ad = env.get_keymap()

pygame.init()
window = pygame.display.set_mode((size, size))
clock = pygame.time.Clock()

def im_postproc(im, a, r):
  d = ImageDraw.Draw(im)
  if (to_pause):
    d.text((10, im.height-100), "PAUSED", (255, 255, 0))
  d.text((10, im.height-80), f"Action: {a}", (255, 255, 255))
  # Time is rendered here by the environment itself.
  rs = ''
  for k, v in r.items():
      rs += f'{k}: {v}'
  d.text((10, im.height-40), f"Reward: {rs}", (255, 255, 255))
  if (args.ckpt_path is not None):
    vs = ''
    for k in o.keys():
        vs += f'{k}: {query_value(agent, o[k], env):.2f} '
    d.text((10, im.height-20), f"Agent Value: {vs}", (255, 255, 255))
  return im

run = True
paused = to_pause = False # Is the simulation paused?
agent_control = 2 if (args.ckpt_path is not None) else 0 # 0: human only 1: human versus AI 2: AI only
pause_key, override_key, restart_key = pygame.K_p, pygame.K_o, pygame.K_r

term = trunc = done = False
a = {k: np.zeros_like(env.action_spaces[k].nvec) for k in env.action_spaces.keys()}
total_rewards = defaultdict(lambda:0)
reset_acts = []
restart = False # restart the environment
while run:
    clock.tick(20)
    # Handle a MultiDiscrete action space
    for k, act in reset_acts:
        a[k][act] = 0 # Reset 'key press' actions
        reset_acts = []
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.KEYDOWN:
            action = event.key
            if (action==pause_key):
                to_pause = True
            elif (action==override_key):
                # Cycle between all, one, and zero human-controlled agents.
                if (args.ckpt_path is not None):
                    agent_control = (agent_control + 1) % 3
            elif (action==restart_key):
                restart = True
            for k in ad.keys():
                if (action in ad[k]): # Is this an action within our environment?
                    x = ad[k][action]
                    a[k][x[0]] = x[1]
                    if (x[2]==True):
                        reset_acts.append((k, x[0]))
        elif event.type == pygame.KEYUP:
            action = event.key
            for k in ad.keys():
                if (action in ad[k]): # Reset
                    x = ad[k][action]
                    a[k][x[0]] = ad[k][DEFAULT_ACTION]
    if ((done or paused) == False):
        if (agent_control >= 1): # If we're running an agent
            ks = sorted(list(a.keys()))
            for i in range(agent_control%2, len(ks)):
                a[ks[i]] = query_model(agent, o[ks[i]], env, env.action_spaces[ks[i]])
        o, r, term, trunc, _ = env.step(a)
        for k, v in r.items():
            total_rewards[k] += v
        # Render
        render_output = env.render()
        render_output = im_postproc(render_output, a, total_rewards)
        raw_str = render_output.tobytes("raw", 'RGBA')
        pygame_surface = pygame.image.fromstring(raw_str, (size, size), 'RGBA')
        window.blit(pygame_surface, (0, 0))
        pygame.display.flip()
    if (to_pause):
        paused = (not paused)
        to_pause = False
    if (restart): # reset on r if game is over
        o, _ = env.reset()
        term=trunc={'__all__':False}
        restart = False
        total_rewards = defaultdict(lambda:0)
    done = term['__all__'] or trunc['__all__']