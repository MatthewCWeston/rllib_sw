import numpy as np
from environments.SpaceWar_constants import *

def wrap(p):
    ''' Wraps a point within a circle of radius 1 and center [0, 0] '''
    dist = (p**2).sum()
    if (dist <= 1):
        return
    else:
        past = dist**.5 - 1
        p *= -1 * (1-past)

class Missile():
    REPR_SIZE = 5
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel
        self.life = self.maxLife = MISSILE_LIFE
    def update(self, ships, speed): # Returns hit_player, remove_self
        mp = self.pos
        mv = self.vel
        mp += mv * speed # Move missile
        for si in range(len(ships)): # Missile hits target
            if (np.linalg.norm(mp-ships[si].pos, 2) < PLAYER_SIZE):
                return si, True
        if (np.linalg.norm(mp, 2) < STAR_SIZE): # Missile hits star
            return -1, True
        wrap(mp)
        self.life -= 1 * speed
        return -1, (self.life<=0)
    def get_obs(self):
        return np.array([self.pos[0], self.pos[1], self.vel[0], self.vel[1], self.life/self.maxLife])

class Ship():
    REPR_SIZE = 8
    def __init__(self, pos, ang):
        self.pos = pos
        self.ang = ang
        self.stored_missiles = NUM_MISSILES
        self.vel = np.array([0.,0.])
        self.reloadTime = 0
        self.updateAngUV()
    def updateAngUV(self):
        a = self.ang*np.pi/180
        self.angUV = np.array([np.cos(a), -np.sin(a)])
    def update(self, action, missiles, speed):
        self.updateAngUV()
        # Take actions 
        if (action[0]==1):
          self.vel += SHIP_ACC * self.angUV * speed
        if (action[1]==1):
          self.ang += SHIP_TURN_RATE * speed
        elif (action[1]==2):
          self.ang -= SHIP_TURN_RATE * speed
        # Shoot 
        if (action[2]==1 and self.stored_missiles > 0 and self.reloadTime <= 0):
            m = Missile(self.pos + self.angUV * PLAYER_SIZE, 
                self.vel + self.angUV * MISSILE_VEL)
            missiles.append(m)
            self.stored_missiles -= 1
            self.reloadTime = MISSILE_RELOAD_TIME
        else:
            self.reloadTime -=  speed
        # Update position
        self.pos += self.vel * speed
        self.vel = np.clip(self.vel, -1.0, 1.0)
        wrap(self.pos)
        # Apply force of gravity. GMm can be treated as a single constant.
        self.vel -= (self.pos * GRAV_CONST / (self.pos[0]**2 + self.pos[1]**2)** 1.5) * speed
    def get_obs(self):
        # pos, vel, angle unit vector, ammo remaining
        return np.concatenate([self.pos, self.vel, self.angUV, 
            [self.stored_missiles / NUM_MISSILES, self.reloadTime / MISSILE_RELOAD_TIME]
            ])