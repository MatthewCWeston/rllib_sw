import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import MultiDiscrete, Box, Dict
import numpy as np
from PIL import Image, ImageDraw
import pygame

from classes.repeated_space import RepeatedCustom

from environments.SpaceWar_constants import *
from environments.SpaceWar_objects import Missile, Ship

class SW_1v1_env(MultiAgentEnv):
	def __init__(self, env_config={}):
		super().__init__()
		self.maxTime = env_config['ep_length'] if 'ep_length' in env_config else DEFAULT_MAX_TIME
		self.speed = env_config['speed'] if 'speed' in env_config else 1.0
		self.size = env_config['render_size'] if 'render_size' in env_config else DEFAULT_RENDER_SIZE
		self.egocentric = env_config['egocentric'] if 'egocentric' in env_config else True
		self.augment_obs = env_config.get('aug_obs', self.egocentric)
		self.random_orbit_prob = env_config.get('random_orbit_prob', 0.0) # For 'curriculum' training
		assert (self.egocentric==True or self.augment_obs==False), "Augmentated observations supported for ego. obs. only"
		self.metadata['render_modes'].append('rgb_array')
		self.render_mode = 'rgb_array'
		# nop/thrust, nop/left/right, npo/shoot
		self.action_spaces = {i: MultiDiscrete([2,3,2]) for i in range(2)}
		# Observation spaces; fixed and variable
		missile_space = Box(-1,1,shape=(Missile.REPR_SIZE+Missile.AUG_DIM*self.augment_obs,))
		self.missile_space = RepeatedCustom(missile_space, NUM_MISSILES)
		self.observation_spaces = {i: Dict({
			"self": Box(-1,1,shape=(Ship.REPR_SIZE+Ship.ALL_AUG_DIM*self.augment_obs,)), 
			"opponent": Box(-1,1,shape=(Ship.REPR_SIZE+Ship.OTHER_AUG_DIM*self.augment_obs,)),
			"missiles_friendly": self.missile_space, # Friendly missiles
			"missiles_hostile": self.missile_space # Hostile missiles
		}) for i in range(2)}
		self.agents = list(self.observation_spaces.keys())
		
	def get_obs(self):
		obs = {}
		for i in range(2):
			ego = self.playerShips[i] if self.egocentric else None
			obs[i] = {
				"self": self.playerShips[i].get_obs(ego, self.augment_obs),
				"opponent": self.playerShips[(i+1)%2].get_obs(ego, self.augment_obs),
				"missiles_friendly": self.missile_space.encode_obs([m.get_obs(ego, self.augment_obs) for m in self.missiles[(i)%2]]),
				"missiles_hostile": self.missile_space.encode_obs([m.get_obs(ego, self.augment_obs) for m in self.missiles[(i+1)%2]]),
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
				
	def random_orbits(self):
		'''
			Generate a pair of opposing orbits, and randomly perturb some other things.
		'''
		position = np.random.uniform(-WRAP_BOUND,WRAP_BOUND, (2,))
		r = np.random.uniform(0.4, 1)**.5 * WRAP_BOUND # Nothing too close, we don't want to waste episodes on trivial outcomes
		p_ang = np.random.uniform(0, 2*np.pi)
		position = np.array([np.cos(p_ang), np.sin(p_ang)]) * r
		dist_zero = (position**2).sum()**.5
		# Set velocity (perpendicular to angle to star)
		g = GRAV_CONST #/ (position[0]**2 + position[1]**2)
		r_p = np.random.uniform(low=max(PLAYER_SIZE*2+STAR_SIZE, r/2), high=r)
		ecc = (r - r_p) / (r + r_p)
		# velocity at apocentre is sqrt((1-e) * GM / ((1+e) * r_a))
		v_magnitude = ((1-ecc) * g / ((1+ecc)*r))**.5
		v_angle = np.arctan2(position[1],position[0]) + np.pi/2 * np.sign(np.random.rand()-0.5)
		# Store velocity and position
		pos_1 = position
		vel_1 = np.array([np.cos(v_angle), np.sin(v_angle)]) * v_magnitude
		facing_1 = np.random.uniform(0, 360)
		pos_2 = -position # Opposite side of the star
		vel_2 = -vel_1
		facing_2 = facing_1 + 180
		ammo = int(NUM_MISSILES * np.random.uniform(0.25,1))
		self.playerShips = [
			Ship(pos_1, facing_1, vel=vel_1),
			Ship(pos_2, facing_2, vel=vel_2)
		]
		for p in self.playerShips:
			p.stored_missiles = ammo
	
	def reset(self, seed=0, options={}):
		
		self.missiles = [[],[]] # x, y, vx, vy
		self.time = 0
		self.terminated = False # for rendering purposes
		self.last_acts = [[0,0,0],[0,0,0]]
		if np.random.rand() < self.random_orbit_prob:
			# Should give agents more diverse experience when training
			self.random_orbits()
		else:
			self.playerShips = [
				Ship(np.array([-.5, .5]), 90.),
				Ship(np.array([.5, -.5]), 270.)
			]
		return self.get_obs(), {}
		
	def step(self, actions):
		self.rewards = {0:0,1:0}
		self.time += 1 * self.speed
		self.last_acts = actions # for rendering
		# Thrust is acc times anguv
		for i, ship, in enumerate(self.playerShips):
			ship.update(actions[i], self.missiles[i], self.speed)
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
		# Early stop when both participants run out of ammo, but only when we're already messing with things to train..
		truncated = (self.time >= self.maxTime) or ((len(self.missiles[0]) + len(self.missiles[1]) == 0) and (self.playerShips[0].stored_missiles == 0) and (self.playerShips[0].stored_missiles == 0) and (self.random_orbit_prob > 0))
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
		for i, ship in enumerate(self.playerShips):
			ship.render(draw, dim, hdim, ssz, self.terminated, reward=self.rewards[i], ego=ego)
		# Draw the missiles
		for m in self.missiles[0]:
			m.render(draw, hdim, msz, self.speed, ego=ego)
		for m in self.missiles[1]:
			m.render(draw, hdim, msz, self.speed, ego=ego, c="orange")
		# Rotate 90 degrees for easier viewing when using egocentric rendering
		if (self.egocentric):
			img = img.rotate(90)
			draw = ImageDraw.Draw(img)
		# Draw the time
		draw.text((10, img.height-60), f"Time: {self.time}/{self.maxTime}", (255, 255, 255))
		return img
