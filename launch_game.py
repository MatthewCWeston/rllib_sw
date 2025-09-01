'''
    Launch an instance of the SPACEWAR RL environment, with a pygame wrapper.
        
    For now, we take a path to an rllib env with:
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

env_name = sys.argv[1]

parser = argparse.ArgumentParser()
parser.add_argument("env-name", type=str, help="Name of the environment")
parser.add_argument("--env-config", type=json.loads, default={})
parser.add_argument("--ckpt-path", type=str, default=None, help="Path to checkpoint file, if applicable")
args = parser.parse_args()

cfg = args.env_config
agent = None
if (args.ckpt_path is not None):
    from classes.inference_helpers import load_checkpoint, query_model, query_value
    ckpt_path = os.path.abspath(args.ckpt_path)
    agent = load_checkpoint(ckpt_path)


def instantiate_env(env_name, cfg):
    spec = importlib.util.spec_from_file_location(env_name, f'./environments/{env_name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_class = getattr(module, env_name)
    return agent_class(cfg)
env = instantiate_env(env_name, cfg)
ad = env.get_keymap()
ma_env = hasattr(env, 'agents')
if (ma_env): # Handle envs in the multi-agent format
    agent_name = env.agents[0]
    ad = ad[agent_name]
o, _ = env.reset()
size = env.size


pygame.init()
window = pygame.display.set_mode((size, size))
clock = pygame.time.Clock()

def im_postproc(im, a, r):
  d = ImageDraw.Draw(im)
  if (to_pause):
    d.text((10, im.height-100), "PAUSED", (255, 255, 0))
  d.text((10, im.height-80), f"Action: {a}", (255, 255, 255))
  # Time is rendered here by the environment itself.
  d.text((10, im.height-40), f"Reward: {r}", (255, 255, 255))
  if (args.ckpt_path is not None):
    v = query_value(agent, o, env)
    d.text((10, im.height-20), f"Agent Value: {v:.2f}", (255, 255, 255))
  return im

run = True
paused = to_pause = False # Is the simulation paused?
agent_control = (args.ckpt_path is not None)
pause_key, override_key = pygame.K_p, pygame.K_o

term = trunc = done = False
action = "default"
total_reward = 0
if ma_env:
    a = np.zeros_like(env.action_space[agent_name].nvec)
else:
    a = np.zeros_like(env.action_space.nvec)
reset_acts = []
while run:
    clock.tick(20)
    # Handle a MultiDiscrete action space
    for act in reset_acts:
        a[act] = 0 # Reset 'key press' actions
        reset_acts = []
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.KEYDOWN:
            action = event.key
            if (action in [pause_key, override_key]):
                if (action==pause_key):
                    to_pause = True
                elif (action==override_key):
                    agent_control = (not agent_control) and (args.ckpt_path is not None)
            elif (action in ad): # Is this an action within our environment?
                x = ad[action]
                a[x[0]] = x[1]
                if (x[2]==True):
                    reset_acts.append(x[0])
        elif event.type == pygame.KEYUP:
            action = event.key
            if (action in ad): # Reset
                x = ad[action]
                a[x[0]] = 0
    if ((done or paused) == False):
        if (agent_control): # If we're running an agent
            a = query_model(agent, o, env)
        if ma_env:
            o, r, term, trunc, _ = env.step({agent_name: a})
            o, r = o[agent_name], r[agent_name]
            term, trunc = term['__all__'], trunc['__all__']
        else:
            o, r, term, trunc, _ = env.step(a)
        total_reward += r
        # Render
        render_output = env.render()
        render_output = im_postproc(render_output, a, total_reward)
        raw_str = render_output.tobytes("raw", 'RGBA')
        pygame_surface = pygame.image.fromstring(raw_str, (size, size), 'RGBA')
        window.blit(pygame_surface, (0, 0))
        pygame.display.flip()
    if (to_pause):
        paused = (not paused)
        to_pause = False
    if (action == pygame.K_r): # reset on r if game is over
        o, _ = env.reset()
        term=trunc=False
        action = "default"
        total_reward = 0
    done = term or trunc