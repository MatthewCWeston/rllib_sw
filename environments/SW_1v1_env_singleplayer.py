import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box, Dict
from ray.rllib.utils.spaces.repeated import Repeated
import numpy as np
from PIL import Image, ImageDraw
import pygame

from environments.SpaceWar_constants import *
from environments.SpaceWar_objects import Missile, Ship

# Add circle wrap; add variable time setting
def wrap(p):
    ''' Wraps a point within a circle of radius 1 and center [0, 0] '''
    dist = (p**2).sum()
    if (dist <= 1):
        return
    else:
        past = dist**.5 - 1
        p *= -1 * (1-past)
        
class Dummy_Ship(Ship):
    def get_obs(self):
        # pos, vel, angle unit vector, ammo remaining
        return np.concatenate([self.pos, np.zeros((Ship.REPR_SIZE-2,))])

class SW_1v1_env_singleplayer(gym.Env):
    def __init__(self, env_config={}):
        # nop/thrust, nop/left/right, npo/shoot
        self.action_space = {i: MultiDiscrete([2,3,2]) for i in range(1)}
        # Observation spaces; fixed and variable
        ship_space = Box(-1,1,shape=(Ship.REPR_SIZE*2,))
        missile_space = Box(-1,1,shape=(Missile.REPR_SIZE,))
        missile_space = Repeated(missile_space, NUM_MISSILES)
        self.observation_space = Dict({
            "self": ship_space, # my ship, enemy ship
            "opponent": ship_space,
            "missiles_friendly": missile_space, # Friendly missiles
            "missiles_hostile": missile_space # Hostile missiles
        })
        self.agents = self.possible_agents = [0,1]
        self.maxTime = env_config['ep_length'] if 'ep_length' in env_config else DEFAULT_MAX_TIME
        self.speed = env_config['speed'] if 'speed' in env_config else 1.0
        self.size = env_config['render_size'] if 'render_size' in env_config else DEFAULT_RENDER_SIZE
        self.metadata['render_modes'].append('rgb_array')
        self.render_mode = 'rgb_array'
    def get_obs(self):
        return {0: {
                "self": self.playerShips[0].get_obs(),
                "opponent": self.playerShips[1].get_obs(),
                "missiles_friendly": [m.get_obs() for m in self.missiles],
                "missiles_hostile":  [],
              }
            }
    def get_keymap(self): # Set multidiscrete 
        return {0: {pygame.K_UP: (0,1,False), # Action, Value, hold_disallowed (qol)
                pygame.K_LEFT: (1,1,False), pygame.K_RIGHT: (1,2,False),
                pygame.K_DOWN: (2,1,False),
                "default": 0},
                }
                
    def new_target_position(self):
        dist = np.random.uniform(.1,1)
        angle = np.random.uniform(0,2*np.pi)
        self.playerShips[1].pos = np.array([np.cos(angle), np.sin(angle)])*dist
    
    def reset(self, seed=0, options={}):
        self.playerShips = [
            Ship(np.array([-.5, .5]), 90.),
            Dummy_Ship(np.array([0.,0.]),0.)
        ]
        self.new_target_position()
        self.missiles = [] # x, y, vx, vy
        self.time = 0
        self.terminated = False # for rendering purposes
        self.last_act = [0,0,0]
        return self.get_obs(), {}
    def step(self, actions):
        self.rewards = {0:0}
        self.time += 1 * self.speed
        self.last_act = actions[0] # for rendering
        # Thrust is acc times anguv
        ship = self.playerShips[0]
        ship.update(actions[0], self.missiles, self.speed)
        if (np.linalg.norm(ship.pos, 2) < PLAYER_SIZE + STAR_SIZE):
            self.terminated = True;
            self.rewards = {0: -1}
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
        dim = self.size
        hdim=dim/2
        psz = PLAYER_SIZE * dim
        ssz = STAR_SIZE * dim
        msz = 3
        img = Image.new('RGB', (dim, dim), color='black')
        draw = ImageDraw.Draw(img)
        # Draw the star
        for i in range(2):
            ss = np.random.uniform(-1.0, 1.0, 2) * ssz/2
            draw.line((hdim+ss[0], hdim+ss[1], hdim-ss[0], hdim-ss[1]), fill='white', width=1)
        # Draw the player
        ship = self.playerShips[0]
        p = (ship.pos+1) * hdim
        ppts = np.array([[psz/2, 0], [-psz*3/8, -psz/4], [-psz/2,0], [-psz*3/8, psz/4]])
        a = ship.ang*np.pi/180
        rm = np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]])
        ppts = [(x[0], x[1]) for x in np.dot(ppts, rm) + p]
        draw.ellipse((p[0]-psz/2, p[1]-psz/2, p[0]+psz/2, p[1]+psz/2), outline='gray' if not self.terminated else 'red')
        draw.polygon(ppts, fill='white')
        # Draw the thruster flare
        if (self.last_act[0]==1):
            ppts = np.array([[-psz*3/4, 0], [-psz*3/8, -psz/4], [-psz/2,0], [-psz*3/8, psz/4]])
            a = ship.ang*np.pi/180
            rm = np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]])
            ppts = [(x[0], x[1]) for x in np.dot(ppts, rm) + p]
            draw.polygon(ppts, fill='orange')
        # Draw the player's ammo counter
        bar_len = psz * 2
        x_offset = p[0] - bar_len / 2
        y = p[1] + psz + 10
        x = x_offset
        draw.line([(x, y), (x+bar_len, y)], width=1, fill='gray')
        draw.line([(x, y), (x+bar_len*ship.stored_missiles/NUM_MISSILES, y)], width=1, fill='white')
        y+=2
        draw.line([(x, y), (x+bar_len*max(0,ship.reloadTime)/MISSILE_RELOAD_TIME, y)], width=1, fill='yellow')
        # Draw the target
        target = self.playerShips[1]
        p = (target.pos+1) * hdim
        draw.ellipse((p[0]-psz/2, p[1]-psz/2, p[0]+psz/2, p[1]+psz/2), outline='yellow')
        # Draw the missiles
        for m in self.missiles:
            sz = msz if m.life > self.speed else msz*4
            m = (m.pos+1) * hdim
            draw.ellipse((m[0]-sz/2, m[1]-sz/2, m[0]+sz/2, m[1]+sz/2), outline='yellow', fill='yellow')
        # Draw the wrapping radius 
        draw.ellipse((0, 0, dim, dim), outline='white')
        # Draw the time
        draw.text((10, img.height-60), f"Time: {self.time}/{self.maxTime}", (255, 255, 255))
        return img
