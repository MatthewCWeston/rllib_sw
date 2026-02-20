import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box, Dict
import numpy as np
from PIL import Image, ImageDraw
import pygame

from classes.repeated_space import RepeatedCustom

from environments.SpaceWar_constants import *
from environments.SpaceWar_objects import Missile, Ship, ego_pt, wrap
      
class Dummy_Ship(Ship):
    def update(self, missiles, speed, grav_multiplier, target_loc):
        self.updateAngUV()
        # Update position
        self.pos += self.vel * speed
        self.vel = np.clip(self.vel, -1.0, 1.0)
        wrap(self.pos)
        # Apply force of gravity. GMm can be treated as a single constant.
        self.vel -= (self.pos * GRAV_CONST / (self.pos[0]**2 + self.pos[1]**2)** 1.5) * speed * grav_multiplier

class SW_lead_target(gym.Env):
    def __init__(self, env_config={}):
        super().__init__()
        self.agents = self.possible_agents = [0]
        # nop/thrust, nop/left/right, npo/shoot
        self.egocentric = env_config.get('egocentric', True) # Egocentric coordinates are just better.
        self.render_egocentric = env_config.get('render_egocentric', False)
        self.maxTime = env_config.get('ep_length', DEFAULT_MAX_TIME)
        self.speed = env_config.get('speed', 1.0)
        self.size = env_config.get('render_size', DEFAULT_RENDER_SIZE)
        # Target size multiplier
        self.size_multiplier = self.true_size_multiplier = env_config.get('size_multiplier', 1.0)
        self.elliptical = env_config.get('elliptical', True)
        # Rendering
        self.metadata['render_modes'].append('rgb_array')
        self.render_mode = 'rgb_array'
        # Observation spaces; fixed and variable
        ship_space = Box(-1,1,shape=(Ship.REPR_SIZE,))
        self.missile_space = Box(-1,1,shape=(Missile.REPR_SIZE,))
        obs_space = {
            "self": ship_space, # my ship, enemy ship
            "opponent": ship_space,
            "missiles_friendly": self.missile_space, # Friendly missiles
        }
        self.observation_spaces = {i: Dict(obs_space) for i in range(1)}
        self.action_spaces = {i: MultiDiscrete([2,2]) for i in range(1)}
        
    def get_obs(self):
        ego = self.playerShips[0] if self.egocentric else None
        m = self.missiles
        obs = {
            "self": self.playerShips[0].get_obs(ego),
            "opponent": self.playerShips[1].get_obs(ego),
            "missiles_friendly": m[0].get_obs(ego) if len(m) > 0 else np.zeros((Missile.REPR_SIZE,)),
        }
        return {0: obs}
                
    def new_target_position(self):
        target = self.playerShips[1]
        position = np.random.uniform(-WRAP_BOUND,WRAP_BOUND, (2,))
        r = np.random.uniform(0, 1)**.5 * WRAP_BOUND
        p_ang = np.random.uniform(0, 2*np.pi)
        position = np.array([np.cos(p_ang), np.sin(p_ang)]) * r
        # Set velocity (perpendicular to angle to star)
        g = GRAV_CONST
        if (not self.elliptical):
            v_magnitude = (g/r)**.5 # For a circular orbit (v^2 = GM/r; g=GM)
        else: #   eccentricity = (r_a - r_p) / (r_a + r_p); r_a = r
            r_p = np.random.uniform(low=max(PLAYER_SIZE*2+STAR_SIZE, r/2), high=r)
            ecc = (r - r_p) / (r + r_p)
            # velocity at apocentre is sqrt((1-e) * GM / ((1+e) * r_a))
            v_magnitude = ((1-ecc) * g / ((1+ecc)*r))**.5
        v_angle = np.arctan2(position[1],position[0]) + np.pi/2 * np.sign(np.random.rand()-0.5)
        target.vel = np.array([np.cos(v_angle), np.sin(v_angle)]) * v_magnitude
        target.pos = position
    
    def reset(self, seed=None, options={}):
        self.playerShips = [
            Ship(np.array([1e-10, 0.]), 90.),
            Dummy_Ship(np.array([0.,0.]),0.,PLAYER_SIZE*self.size_multiplier)
        ]
        self.playerShips[0].stored_missiles = 1
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
        actions = [
            0, # Thrust (none)
            actions[0][0]+1, # Turn (right or left only)
            actions[0][1]    # shoot (nop or fire)
        ]
        # If we have a missile already, loop until it resolves.
        target = self.playerShips[1]
        ms = self.missiles
        while(len(ms)>0):
            target.update([], self.speed, grav_multiplier=1, target_loc=ship.pos)
            si,d = ms[0].update(self.playerShips, self.speed)
            if (d):
                del ms[0]
            if (si != -1):
                self.terminated = True
                self.rewards[0] = 1
        ship.update(actions, self.missiles, self.speed, grav_multiplier=0)
        # Update the dummy ship
        target.update([], self.speed, 
            grav_multiplier=1, target_loc=ship.pos)
        if (np.linalg.norm(target.pos, 2) < PLAYER_SIZE + STAR_SIZE):
            self.new_target_position() # If it crashes, respawn it
        # Update missiles
        for i in reversed(range(len(ms))):
            si,d = ms[i].update(self.playerShips, self.speed) # Return hit_obj
            if (d):
                del ms[i]
            if (si != -1):
                self.terminated = True
                self.rewards[0] = 1
        if (ship.stored_missiles == 0 and len(ms)==0):
            self.terminated = True # End environment if we missed.
        truncated = (self.time >= self.maxTime)
        return self.get_obs(), self.rewards, {"__all__": self.terminated}, {"__all__": truncated}, {}