import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.spaces.repeated import Repeated
from gymnasium.spaces import MultiDiscrete, Box, Dict
import numpy as np
from PIL import Image, ImageDraw
import pygame

from classes.repeated_space import RepeatedCustom

from environments.SpaceWar_constants import *
from environments.SpaceWar_objects import Missile, Ship

class SW_1v1_env(MultiAgentEnv):
    def __init__(self, env_config={}):
        # nop/thrust, nop/left/right, npo/shoot
        self.action_spaces = {i: MultiDiscrete([2,3,2]) for i in range(2)}
        # Observation spaces; fixed and variable
        ship_space = Box(-1,1,shape=(Ship.REPR_SIZE,))
        missile_space = Box(-1,1,shape=(Missile.REPR_SIZE,))
        self.missile_space = RepeatedCustom(missile_space, NUM_MISSILES)
        self.observation_spaces = Dict({
            "self": ship_space, # my ship, enemy ship
            "opponent": ship_space,
            "missiles_friendly": self.missile_space, # Friendly missiles
            "missiles_hostile": self.missile_space # Hostile missiles
        })
        self.maxTime = env_config['ep_length'] if 'ep_length' in env_config else DEFAULT_MAX_TIME
        self.speed = env_config['speed'] if 'speed' in env_config else 1.0
        self.size = env_config['render_size'] if 'render_size' in env_config else DEFAULT_RENDER_SIZE
        self.egocentric = env_config['egocentric'] if 'egocentric' in env_config else False
        self.metadata['render_modes'].append('rgb_array')
        self.render_mode = 'rgb_array'
    def get_obs(self):
        obs = {}
        for i in range(2):
            ego = self.playerShips[i] if self.egocentric else None
            obs[i] = {
                "self": self.playerShips[i].get_obs(ego),
                "opponent": self.playerShips[(i+1)%2].get_obs(ego),
                "missiles_friendly": self.missile_space.encode_obs([m.get_obs(ego) for m in self.missiles[(i+1)%2]]),
                "missiles_hostile": self.missile_space.encode_obs([]),
              }
        return obs
    def get_keymap(self): # Set multidiscrete 
        return {0: {pygame.K_UP: (0,1,False), # Action, Value, hold_disallowed (qol)
                pygame.K_LEFT: (1,1,False), pygame.K_RIGHT: (1,2,False),
                pygame.K_DOWN: (2,1,False),
                "default": 0},
                1: {pygame.K_w: (0,1,False), # Action, Value, hold_disallowed (qol)
                pygame.K_a: (1,1,False), pygame.K_d: (1,2,False),
                pygame.K_s: (2,1,False),
                "default": 0},
                }
    
    def reset(self, seed=0, options={}):
        self.playerShips = [
            Ship(np.array([-.5, .5]), 90.),
            Ship(np.array([.5, -.5]), 270.)
        ]
        self.missiles = [[],[]] # x, y, vx, vy
        self.time = 0
        self.terminated = False # for rendering purposes
        self.last_acts = [[0,0,0],[0,0,0]]
        return self.get_obs(), {}
    def step(self, actions):
        self.rewards = {0:0,1:0}
        self.time += 1 * self.speed
        self.last_acts = actions # for rendering
        # Thrust is acc times anguv
        for i, ship, in enumerate(self.playerShips):
            ship.update(actions[i], self.missiles, self.speed)
            if (np.linalg.norm(ship.pos, 2) < PLAYER_SIZE + STAR_SIZE):
                self.terminated = True;
                self.rewards[i] = -1
                self.rewards[(i+1)%2] = 1
        # Update missiles
        for m in self.missiles:
            for i in reversed(range(len(m))):
                si,d = m[i].update(self.playerShips, self.speed) # Return hit_obj
                if (d):
                    del m[i]
                if (si != -1):
                    self.terminated = True
                    self.rewards[si] = -1
                    self.rewards[(si+1)%2] = 1
        truncated = (self.time >= self.maxTime)
        return self.get_obs(), self.rewards, {"__all__": self.terminated}, {"__all__": truncated}, {}
    def render(self): # Display the environment state
        ego = None
        dim = self.size
        hdim=dim/2
        ssz = STAR_SIZE * dim
        msz = 3
        img = Image.new('RGB', (dim, dim), color='black')
        draw = ImageDraw.Draw(img)
        # Draw the star (drawn by player ship if egocentric rendering is active)
        if (ego is None):
            for i in range(2):
                ss = np.random.uniform(-1.0, 1.0, 2) * ssz/2
                draw.line((hdim+ss[0], hdim+ss[1], hdim-ss[0], hdim-ss[1]), fill='white', width=1)
            # Draw the wrapping radius 
            rs = (1-WRAP_BOUND) * hdim
            draw.rectangle((rs,rs,dim-rs,dim-rs), outline='white')
        # Draw the player
        for ship in self.playerShips:
            ship.render(draw, dim, hdim, ssz, self.terminated, ego=ego)
        # Draw the missiles
        for m in self.missiles:
            m.render(draw, hdim, msz, self.speed, ego=ego)
        # Rotate 90 degrees for easier viewing when using egocentric rendering
        if (self.egocentric):
            img = img.rotate(90)
            draw = ImageDraw.Draw(img)
        # Draw the time
        draw.text((10, img.height-60), f"Time: {self.time}/{self.maxTime}", (255, 255, 255))
        return img
