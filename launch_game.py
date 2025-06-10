'''
    Launch an instance of the SPACEWAR RL environment, with a pygame wrapper.
        
    For now, we take a path to an rllib env with:
     - a render() function that returns a PIL image.
     - a getKeyMap() function that maps key inputs to actions within the environment.
'''

import sys
from PIL import ImageDraw
import pygame
import importlib.util
import numpy as np
import json

env_name = sys.argv[1]
if (len(sys.argv) > 2):
    cfg = json.loads(sys.argv[2])
else:
    cfg = {}

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
  d.text((10, im.height-40), f"Action: {a}", (255, 255, 255))
  d.text((10, im.height-20), f"Reward: {r}", (255, 255, 255))
  return im

run = True
term = trunc = done = False
action = "default"
total_reward = 0
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
            if (action in ad):
                x = ad[action]
                a[x[0]] = x[1]
                if (x[2]==True):
                    reset_acts.append(x[0])
        elif event.type == pygame.KEYUP:
            action = event.key
            if (action in ad): # Reset
                x = ad[action]
                a[x[0]] = 0
    if (done == False):
        o, r, term, trunc, _ = env.step(a)
        total_reward += r
        # Render
        render_output = env.render()
        render_output = im_postproc(render_output, a, total_reward)
        raw_str = render_output.tobytes("raw", 'RGBA')
        pygame_surface = pygame.image.fromstring(raw_str, (size, size), 'RGBA')
        window.blit(pygame_surface, (0, 0))
        pygame.display.flip()
    else:
        if (action == pygame.K_r): # reset on r if game is over
            o, _ = env.reset()
            term=trunc=False
            action = "default"
            total_reward = 0
    done = term or trunc