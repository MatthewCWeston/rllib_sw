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
        
def rotate_pt(p, a):
    a = -a*np.pi/180
    x, y = p
    sa = np.sin(a)
    ca = np.cos(a)
    return np.array([x*ca-y*sa,y*ca+x*sa])
    
def ego_pt(p, ego): # egocentric coordinates, accounting for wrapping around a unit circle
    diff = p - ego.pos # relative position 
    pos_adj = rotate_pt(diff, -ego.ang) # Rotate s.t. ego has angle zero
    return pos_adj
    

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
    def get_obs(self, ego):
        if (ego is None):
            return np.array([self.pos[0], self.pos[1], self.vel[0], self.vel[1], self.life/self.maxLife])
        else:
            p = ego_pt(self.pos, ego)
            v = rotate_pt(self.vel, -ego.ang) # Rotate velocity w/r to removing ego's angle
            # rotate relative position by the negative of the angle
            return np.concatenate([p, v, [self.life/self.maxLife]])
    def render(self, draw, hdim, msz, envspeed, ego=None):
        if (ego is None):
            pos, vel = self.pos, self.vel
        else:
            obs = self.get_obs(ego)
            pos, vel = obs[:2], obs[2:4]
        sz = msz if self.life > envspeed else msz*4
        m = (pos+1) * hdim
        draw.ellipse((m[0]-sz/2, m[1]-sz/2, m[0]+sz/2, m[1]+sz/2), outline='yellow', fill='yellow')
        # Draw a cyan line indicating velocity 
        v = vel*msz*hdim*3
        draw.line([m[0],m[1], m[0]+v[0],m[1]+v[1]], width=1, fill='cyan')

class Ship():
    REPR_SIZE = 8
    def __init__(self, pos, ang):
        self.pos = pos
        self.ang = ang
        self.stored_missiles = NUM_MISSILES
        self.vel = np.array([0.,0.])
        self.reloadTime = 0
        self.last_act = [0,0,0]
        self.updateAngUV()
    def updateAngUV(self):
        a = self.ang*np.pi/180
        self.angUV = np.array([np.cos(a), -np.sin(a)])
    def update(self, action, missiles, speed):
        self.updateAngUV()
        self.last_act = action # for rendering
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
    def get_obs(self, ego=None):
        # pos, vel, angle unit vector, ammo remaining
        if (ego is None):
            return np.concatenate([self.pos, self.vel, self.angUV, 
            [self.stored_missiles / NUM_MISSILES, self.reloadTime / MISSILE_RELOAD_TIME]
            ])
        else:
            if (ego==self):
                p = rotate_pt(-self.pos, -ego.ang)# Location of star, adjusted for angle
                auv = np.array([0,0]) # Since we adjust everything else to nullify player angle
            else:
                p = ego_pt(self.pos, ego)
                auv = rotate_pt(self.angUV, -ego.ang)
            v = rotate_pt(self.vel, -ego.ang) # Rotate velocity w/r to removing ego's angle
            return np.concatenate([p, v, auv, 
            [self.stored_missiles / NUM_MISSILES, self.reloadTime / MISSILE_RELOAD_TIME]
            ])
    def render(self, draw,
                dim, hdim, psz, ssz, terminated, ego=None):
        ''' Render this object using draw, adjust for ego if needed.'''
        if (ego is None):
            pos, vel, auv = self.pos, self.vel, self.angUV
            ang = self.ang
        else:
            adj = self.get_obs(ego)
            pos, vel, auv = adj[:2],adj[2:4],adj[4:6]
            ang = self.ang-ego.ang
            if (ego==self):
                # Draw star and wrap boundary
                p = (pos+1)*hdim
                for i in range(2):
                    ss = np.random.uniform(-1.0, 1.0, 2) * ssz/2
                    draw.line((p[0]+ss[0], p[1]+ss[1], p[0]-ss[0], p[1]-ss[1]), fill='white', width=1)
                draw.ellipse([p[0]-hdim,p[1]-hdim,p[0]+hdim,p[1]+hdim], outline='white')
                # Ship should be drawn in the center 
                pos = np.array([0,0])
        
        p = (pos+1) * hdim
        ppts = np.array([[psz/2, 0], [-psz*3/8, -psz/4], [-psz/2,0], [-psz*3/8, psz/4]])
        a = ang*np.pi/180
        rm = np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]])
        ppts = [(x[0], x[1]) for x in np.dot(ppts, rm) + p]
        draw.ellipse((p[0]-psz/2, p[1]-psz/2, p[0]+psz/2, p[1]+psz/2), outline='gray' if not terminated else 'red')
        draw.polygon(ppts, fill='white')
        # Draw the thruster flare
        if (self.last_act[0]==1):
            ppts = np.array([[-psz*3/4, 0], [-psz*3/8, -psz/4], [-psz/2,0], [-psz*3/8, psz/4]])
            a = ang*np.pi/180
            rm = np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]])
            ppts = [(x[0], x[1]) for x in np.dot(ppts, rm) + p]
            draw.polygon(ppts, fill='orange')
        # Draw the player's ammo counter
        bar_len = psz * 2
        x_offset = p[0] - bar_len / 2
        y = p[1] + psz + 10
        x = x_offset
        draw.line([(x, y), (x+bar_len, y)], width=1, fill='gray')
        draw.line([(x, y), (x+bar_len*self.stored_missiles/NUM_MISSILES, y)], width=1, fill='white')
        y+=2
        draw.line([(x, y), (x+bar_len*max(0,self.reloadTime)/MISSILE_RELOAD_TIME, y)], width=1, fill='yellow')
        # Draw an orange line indicating angle unit vector 
        #if (ego is not self):
        auv = auv*psz
        draw.line([p[0],p[1], p[0]+auv[0],p[1]+auv[1]], width=1, fill='orange')
        # Draw a cyan line indicating velocity
        vel = vel * psz*dim*2
        draw.line([p[0],p[1], p[0]+vel[0],p[1]+vel[1]], width=1, fill='cyan')