import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box, Dict
from ray.rllib.utils.spaces.repeated import Repeated
from collections import OrderedDict
import numpy as np
from PIL import Image, ImageDraw
import pygame

# Add circle wrap; add variable time setting
def wrap(p):
    ''' Wraps a point within a circle of radius 1 and center [0, 0] '''
    dist = (p**2).sum()
    if (dist <= 1):
        return
    else:
        past = dist**.5 - 1
        p *= -1 * (1-past)

class SW_Missle():
    def __init__(self, pos, vel, maxLife):
        self.pos = pos
        self.vel = vel
        self.life = self.maxLife = maxLife
    def update(self, goals, playerLoc, env): # Returns hit_goal, hit_player, remove_self
        mp = self.pos
        mv = self.vel
        mp += mv * env.speed # Move missile
        for gi in reversed(range(len(goals))): # Missile hits target
            if (np.linalg.norm(mp-goals[gi], 2) < env.gSize):
                return gi, False, True
        if (np.linalg.norm(mp, 2) < env.sSize): # Missile hits star
            return -1, False, True
        if (np.linalg.norm(playerLoc - mp, 2) < env.pSize): # Missile hits player
            return -1, True, True
        wrap(mp)
        self.life -= 1 * env.speed
        return -1, False, (self.life<=0)
    def get_obs(self):
        return np.array([self.pos[0], self.pos[1], self.vel[0], self.vel[1], self.life/self.maxLife])

# The agent must fly over several goal positions
class SW_MultiShoot_Env(gym.Env):
    def __init__(self, env_config={}):
        self.action_space = MultiDiscrete([2,3,2]) # nop/thrust, nop/left/right, npo/shoot
        # Observation spaces; fixed and variable
        self.num_goals = 5
        self.num_missiles = 5
        general_space = Box(-1,1,shape=(7,)) # [posX, posY, velX, velY, angX, angY, ammo]
        goal_space = Box(-1,1,shape=(2,)) # x, y in range -1, 1
        goal_space = Repeated(goal_space, self.num_goals)
        missile_space = Box(-1,1,shape=(5,)) # x, y, vx, vy, life in range -1, 1
        missile_space = Repeated(missile_space, self.num_missiles)
        self.observation_space = Dict({
            "general": general_space,
            "goals": goal_space,
            "missiles": missile_space
        })
        self.acc = 0.0005 # Rate of player acceleration
        self.missileVel = 0.02 # Initial missile velocity
        self.missileLife = 50 # Missile lifespan
        self.turnRate = 5
        self.maxTime = env_config['ep_length'] if 'ep_length' in env_config else 256
        self.speed = env_config['speed'] if 'speed' in env_config else 1.0
        self.size = 300
        self.pSize = 0.05
        self.sSize = 0.05
        self.gSize = 0.10
        self.gConst = .0001
        self.metadata['render_modes'].append('rgb_array')
        self.render_mode = 'rgb_array'
    def get_obs(self):
        ammo_stock = self.stored_missiles / self.num_missiles
        supplemental = np.array([ammo_stock])
        return OrderedDict([
            ("general", np.concatenate([self.pos, self.vel, self.angUV, supplemental], dtype=np.float32)),
            ("goals", self.goals),
            ("missiles", [m.get_obs() for m in self.missiles])
          ])
    def get_keymap(self): # Set multidiscrete 
        return {pygame.K_UP: (0,1,False), # Action, Value, hold_disallowed (qol)
                pygame.K_LEFT: (1,1,False), pygame.K_RIGHT: (1,2,False),
                pygame.K_SPACE: (2,1,True),
                "default": 0}
    def updateAngUV(self):
        a = self.ang*np.pi/180
        self.angUV = np.array([np.cos(a), -np.sin(a)])
    def reset(self, seed=0, options={}):
        self.pos = np.array([-0.75, -0.75])
        self.vel = np.zeros_like(self.pos)
        self.ang = 30.
        self.updateAngUV()
        self.goals = []
        for _ in range(self.num_goals):
            dist = np.random.uniform(0,1)
            angle = np.random.uniform(0,2*np.pi)
            self.goals.append(np.array([np.cos(angle), np.sin(angle)])*dist)
        self.missiles = [] # x, y, vx, vy
        self.stored_missiles = self.num_missiles
        self.time = 0
        self.terminated = False # for rendering purposes
        self.last_act = 0
        return self.get_obs(), {}
    def step(self, action):
        reward = 0
        self.time += 1 * self.speed
        self.last_act = action # for rendering
        # Thrust is acc times anguv
        if (action[0]==1):
          self.vel += self.acc * self.angUV * self.speed
        # Turn
        if (action[1]==1):
          self.ang += self.turnRate * self.speed
        elif (action[1]==2):
          self.ang -= self.turnRate * self.speed
        self.updateAngUV()
        # Shoot 
        if (action[2]==1 and self.stored_missiles > 0):
            m = SW_Missle(self.pos + self.angUV * self.pSize, 
                self.vel + self.angUV * self.missileVel, 
                self.missileLife)
            self.missiles.append(m)
            self.stored_missiles -= 1
        # Apply force of gravity. GMm can be treated as a single constant.
        self.vel -= (self.pos * self.gConst / (self.pos[0]**2 + self.pos[1]**2)** 1.5) * self.speed
        # Update missiles
        for i in reversed(range(len(self.missiles))):
            hg,hp,d = self.missiles[i].update(self.goals, self.pos, self) # Return hit_obj
            if (d):
                del self.missiles[i]
            if (hp):
                self.terminated = True
                reward -= 10
            if (hg != -1):
                del self.goals[hg]
                reward += 10
        # Update
        self.pos += self.vel * self.speed
        self.vel = np.clip(self.vel, -1.0, 1.0)
        # Wrap
        wrap(self.pos)
        # Check for collision with star
        # Reward is +10 for hitting a target, -10 for collidijng with the star or a missile.
        if (np.linalg.norm(self.pos, 2) < self.pSize + self.sSize):
            self.terminated = True;
            reward -= 10
        '''elif (len(self.missiles) == 0 and self.stored_missiles==0):
            self.terminated = True # End game when no missiles stored or in flight'''
        if (self.terminated==False):
            self.terminated=(len(self.goals)==0)
        truncated = (self.time >= self.maxTime)
        return self.get_obs(), reward, self.terminated, truncated, {}
    def render(self): # Display the environment state
        dim = self.size
        hdim=dim/2
        p = (self.pos+1) * hdim
        psz = self.pSize * dim
        ssz = self.sSize * dim
        gsz = self.gSize * dim
        msz = 3
        img = Image.new('RGB', (dim, dim), color='black')
        draw = ImageDraw.Draw(img)
        # Draw the star
        for i in range(2):
            ss = np.random.uniform(-1.0, 1.0, 2) * ssz/2
            draw.line((hdim+ss[0], hdim+ss[1], hdim-ss[0], hdim-ss[1]), fill='white', width=1)
        # Draw the player (circle with line indicating direction) and the goal
        pCol = 'white' if not self.terminated else 'red'
        ppts = np.array([[psz/2, 0], [-psz*3/8, -psz/4], [-psz/2,0], [-psz*3/8, psz/4]])
        a = self.ang*np.pi/180
        rm = np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]])
        ppts = [(x[0], x[1]) for x in np.dot(ppts, rm) + p]
        draw.ellipse((p[0]-psz/2, p[1]-psz/2, p[0]+psz/2, p[1]+psz/2), outline='gray')
        draw.polygon(ppts, fill=pCol)
        # Draw the thruster flare
        if (self.last_act[0]==1):
            ppts = np.array([[-psz*3/4, 0], [-psz*3/8, -psz/4], [-psz/2,0], [-psz*3/8, psz/4]])
            a = self.ang*np.pi/180
            rm = np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]])
            ppts = [(x[0], x[1]) for x in np.dot(ppts, rm) + p]
            draw.polygon(ppts, fill='orange')
        # Draw the player's ammo counter
        per_offset = 3
        x_offset = p[0] - self.num_missiles * per_offset / 2
        y = p[1] + psz + 10
        for i in range(self.num_missiles):
            x = x_offset + i*per_offset
            c = 'white' if self.stored_missiles > i else 'gray'
            draw.line([(x, y), (x, y + 10)], width=1, fill=c)
        # Draw the goals
        for g in self.goals:
            g = (g+1) * hdim
            draw.ellipse((g[0]-gsz/2, g[1]-gsz/2, g[0]+gsz/2, g[1]+gsz/2), outline='green')
        # Draw the missiles
        for m in self.missiles:
            sz = msz if m.life > 1 else msz*4
            m = (m.pos+1) * hdim
            draw.ellipse((m[0]-sz/2, m[1]-sz/2, m[0]+sz/2, m[1]+sz/2), outline='orange',fill='orange')
        # Draw the wrapping radius 
        draw.ellipse((0, 0, dim, dim), outline='white')
        # Draw the time
        draw.text((10, img.height-60), f"Time: {self.time}/{self.maxTime}", (255, 255, 255))
        return img
