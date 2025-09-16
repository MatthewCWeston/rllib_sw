import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.spaces.repeated import Repeated
from gymnasium.spaces import MultiDiscrete, Box, Dict
import numpy as np
from PIL import Image, ImageDraw
import pygame

from classes.repeated_space import RepeatedCustom

from environments.SpaceWar_constants import *
from environments.SpaceWar_objects import Missile, Ship, ego_pt
      
class Dummy_Ship(Ship):
    def get_obs(self, ego=None):
        if (ego is None):
            pos = self.pos
        else:
            pos = ego_pt(self.pos, ego)
        # pos, vel, angle unit vector, ammo remaining
        return np.concatenate([pos, np.zeros((Ship.REPR_SIZE-2,))])
    def render(self, draw, hdim, psz, ego=None):
        if (ego is None):
            pos = self.pos
        else:
            pos = self.get_obs(ego=ego)[:2]
        p = (pos+1) * hdim
        draw.ellipse((p[0]-psz/2, p[1]-psz/2, p[0]+psz/2, p[1]+psz/2), outline='yellow')

class SW_1v1_env_singleplayer(MultiAgentEnv):
    def __init__(self, env_config={}):
        super().__init__()
        self.agents = self.possible_agents = [0]
        # nop/thrust, nop/left/right, npo/shoot
        self.action_spaces = {i: MultiDiscrete([2,3,2]) for i in range(1)}
        # Observation spaces; fixed and variable
        ship_space = Box(-1,1,shape=(Ship.REPR_SIZE,))
        missile_space = Box(-1,1,shape=(Missile.REPR_SIZE,))
        self.missile_space = RepeatedCustom(missile_space, NUM_MISSILES)
        self.missile_space = RepeatedCustom(missile_space, NUM_MISSILES)
        self.observation_spaces = {i: Dict({
            "self": ship_space, # my ship, enemy ship
            "opponent": ship_space,
            "missiles_friendly": self.missile_space, # Friendly missiles
            "missiles_hostile": self.missile_space # Hostile missiles
        }) for i in range(1)}
        self.egocentric = env_config['egocentric'] if 'egocentric' in env_config else False
        self.maxTime = env_config['ep_length'] if 'ep_length' in env_config else DEFAULT_MAX_TIME
        self.speed = env_config['speed'] if 'speed' in env_config else 1.0
        self.size = env_config['render_size'] if 'render_size' in env_config else DEFAULT_RENDER_SIZE
        self.metadata['render_modes'].append('rgb_array')
        self.render_mode = 'rgb_array'
    def get_obs(self):
        ego = self.playerShips[0] if self.egocentric else None
        return {0: {
                "self": self.playerShips[0].get_obs(ego),
                "opponent": self.playerShips[1].get_obs(ego),
                "missiles_friendly": self.missile_space.encode_obs([m.get_obs(ego) for m in self.missiles]),
                "missiles_hostile":  self.missile_space.encode_obs([]),
              }
            }
    def get_keymap(self): # Set multidiscrete 
        return {0: {pygame.K_UP: (0,1,False), # Action, Value, hold_disallowed (qol)
                pygame.K_LEFT: (1,1,False), pygame.K_RIGHT: (1,2,False),
                pygame.K_DOWN: (2,1,False),
                "default": 0},
                }
                
    def new_target_position(self):
        position = np.random.uniform(-WRAP_BOUND,WRAP_BOUND, (2,))
        self.playerShips[1].pos = position
    
    def reset(self, seed=None, options={}):
        self.playerShips = [
            Ship(np.array([-.5, .5]), 90.),
            Dummy_Ship(np.array([0.,0.]),0.)
        ]
        self.new_target_position()
        self.missiles = [] # x, y, vx, vy
        self.time = 0
        self.terminated = False # for rendering purposes
        return self.get_obs(), {}
    def step(self, actions):
        self.rewards = {0:0}
        self.time += 1 * self.speed
        # Thrust is acc times anguv
        ship = self.playerShips[0]
        ship.update(actions[0], self.missiles, self.speed)
        if (np.linalg.norm(ship.pos, 2) < PLAYER_SIZE + STAR_SIZE):
            self.terminated = True;
            self.rewards[0] = -5 # Add a strong incentive to learn not hitting the star early.
        # Update missiles
        for i in reversed(range(len(self.missiles))):
            si,d = self.missiles[i].update(self.playerShips, self.speed) # Return hit_obj
            if (d):
                del self.missiles[i]
            if (si != -1):
                if (si == 0):
                    self.terminated = True
                    self.rewards[0] = -1
                else:
                    self.rewards[0] = 1
                    self.new_target_position()
        truncated = (self.time >= self.maxTime)
        return self.get_obs(), self.rewards, {"__all__": self.terminated}, {"__all__": truncated}, {}
    def render(self): # Display the environment state
        ego = self.playerShips[0] if self.egocentric else None
        dim = self.size
        hdim=dim/2
        psz = PLAYER_SIZE * dim
        ssz = STAR_SIZE * dim
        msz = 3
        img = Image.new('RGB', (dim, dim), color='black')
        draw = ImageDraw.Draw(img)
        # Draw the star (drawn by player ship if egocentric rendering is active)
        if (self.egocentric==False):
            for i in range(2):
                ss = np.random.uniform(-1.0, 1.0, 2) * ssz/2
                draw.line((hdim+ss[0], hdim+ss[1], hdim-ss[0], hdim-ss[1]), fill='white', width=1)
            # Draw the wrapping radius 
            #draw.ellipse((0, 0, dim, dim), outline='white')
            rs = (1-WRAP_BOUND) * hdim
            draw.rectangle((rs,rs,dim-rs,dim-rs), outline='white')
        # Draw the player
        ship = self.playerShips[0]
        ship.render(draw, dim, hdim, psz, ssz, self.terminated, ego=ego)
        # Draw the target
        target = self.playerShips[1]
        target.render(draw, hdim, psz, ego)
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
