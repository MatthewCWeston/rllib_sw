import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box, Dict
import numpy as np
from PIL import Image, ImageDraw
import pygame

from classes.repeated_space import RepeatedCustom

from environments.SpaceWar_constants import *
from environments.SpaceWar_objects import Missile, Ship, ego_pt, wrap

# Augmented observation sizes
SELF_OBS_SIZE = Ship.REPR_SIZE + 4     # 8 + 4 = 12
OPP_OBS_SIZE = Ship.REPR_SIZE + 7     # 8 + 7 = 15
AUG_MISSILE_SIZE = Missile.REPR_SIZE + 3  # 5 + 3 = 8

BASE_REWARD = 'base_reward'


class Dummy_Ship(Ship):
    # ... (unchanged)
    def update(self, missiles, speed, grav_multiplier, target_loc):
        self.updateAngUV()
        if (self.stored_missiles > 0):
            vec_diff = target_loc - self.pos
            wrap(vec_diff)
            ang_diff = (np.arctan2(-vec_diff[1], vec_diff[0]) * 180 / np.pi - self.ang) % 360
            if (ang_diff > 180):
                action = 2
            elif (ang_diff < -180):
                action = 1
            elif (ang_diff > 0):
                action = 1
            else:
                action = 2
            if (action == 1):
                self.ang += SHIP_TURN_RATE * speed
            elif (action == 2):
                self.ang -= SHIP_TURN_RATE * speed
            if (self.stored_missiles > 0 and self.reloadTime <= 0
                    and (vec_diff ** 2).sum() ** .5 < MISSILE_LIFE * MISSILE_VEL * 1.5):
                m = Missile(self.pos + self.angUV * self.size, self.vel + self.angUV * MISSILE_VEL)
                missiles.append(m)
                self.stored_missiles -= 1
                self.reloadTime = MISSILE_RELOAD_TIME
            else:
                self.reloadTime = max(self.reloadTime - speed, 0)
        self.pos += self.vel * speed
        self.vel = np.clip(self.vel, -1.0, 1.0)
        wrap(self.pos)
        self.vel -= (self.pos * GRAV_CONST / (self.pos[0] ** 2 + self.pos[1] ** 2) ** 1.5) * speed * grav_multiplier

    def render(self, draw, hdim, ego=None):
        super().render(draw, hdim * 2, hdim, self.size, False, ego=ego)
        psz = self.size * hdim * 2
        if (ego is None):
            pos = self.pos
        else:
            pos = self.get_obs(ego=ego)[:2]
        p = (pos + 1) * hdim
        draw.ellipse((p[0] - psz / 2, p[1] - psz / 2, p[0] + psz / 2, p[1] + psz / 2), outline='yellow')
        if ((self.vel != 0).any()):
            vel = self.vel * psz * hdim * 5
            draw.line([p[0], p[1], p[0] + vel[0], p[1] + vel[1]], width=1, fill='cyan')
            draw.line([p[0], p[1], hdim, hdim], width=1, fill='green')


class SW_1v1_env_singleplayer(gym.Env):
    def __init__(self, env_config={}):
        super().__init__()
        self.agents = self.possible_agents = [0]
        self.egocentric = env_config.get('egocentric', True)
        self.render_egocentric = env_config.get('render_egocentric', False)
        self.maxTime = env_config.get('ep_length', DEFAULT_MAX_TIME)
        self.speed = env_config.get('speed', 1.0)
        self.size = env_config.get('render_size', DEFAULT_RENDER_SIZE)
        self.grav_multiplier = env_config.get('grav_multiplier', 1.0)
        self.size_multiplier = env_config.get('size_multiplier', 1.0)
        self.target_speed = env_config.get('target_speed', 0.0)
        self.elliptical = env_config.get('elliptical', True)
        self.target_ammo = env_config.get('target_ammo', 0.0)
        self.metadata['render_modes'].append('rgb_array')
        self.render_mode = 'rgb_array'

        # Observation spaces sized for augmented features
        if self.egocentric:
            self_space = Box(-3, 3, shape=(SELF_OBS_SIZE,), dtype=np.float32)
            opp_space = Box(-3, 3, shape=(OPP_OBS_SIZE,), dtype=np.float32)
            self.missile_space = RepeatedCustom(
                Box(-3, 3, shape=(AUG_MISSILE_SIZE,), dtype=np.float32),
                NUM_MISSILES
            )
        else:
            self_space = Box(-1, 1, shape=(Ship.REPR_SIZE,), dtype=np.float32)
            opp_space = Box(-1, 1, shape=(Ship.REPR_SIZE,), dtype=np.float32)
            self.missile_space = RepeatedCustom(
                Box(-1, 1, shape=(Missile.REPR_SIZE,), dtype=np.float32),
                NUM_MISSILES
            )

        obs_space = {
            "self": self_space,
            "opponent": opp_space,
            "missiles_friendly": self.missile_space,
            "missiles_hostile": self.missile_space,
        }
        self.observation_spaces = {i: Dict(obs_space) for i in range(1)}
        self.action_spaces = {i: MultiDiscrete([2, 3, 2]) for i in range(1)}

    def _augment_self_obs(self, self_raw, vel_ego):
        """
        Derive orbital context features from the ego ship's raw observation.
        
        Raw layout (egocentric): [star_pos(2), vel(2), raycasts(2), ammo(1), reload(1)]
        
        Added features:
          - dist_to_star:       scalar distance to gravity well (network can't easily
                                compute norm from rotated Cartesian star position)
          - speed:              velocity magnitude (same problem)
          - radial_vel_star:    velocity component toward/away from star
                                positive = moving toward star = danger
          - tangential_speed:   velocity component perpendicular to star direction
                                indicates orbital motion vs radial fall
        """
        star_pos_ego = self_raw[0:2]

        dist_to_star = np.linalg.norm(star_pos_ego)
        speed = np.linalg.norm(vel_ego)

        if dist_to_star > 1e-8:
            star_dir = star_pos_ego / dist_to_star
            radial_vel_star = np.dot(vel_ego, star_dir)
            tangential_speed = np.linalg.norm(
                vel_ego - radial_vel_star * star_dir
            )
        else:
            radial_vel_star = 0.0
            tangential_speed = speed

        derived = np.array([
            dist_to_star / WRAP_BOUND,      # ~[0, 1.4]
            speed,                           # ~[0, 1.4]
            radial_vel_star,                 # ~[-1.4, 1.4]
            tangential_speed,                # ~[0, 1.4]
        ], dtype=np.float32)

        return np.concatenate([self_raw, derived])

    def _augment_opp_obs(self, opp_raw, vel_ego):
        """
        Derive engagement geometry features from the opponent's raw observation.
        
        Raw layout (egocentric): [rel_pos(2), rel_vel(2), angUV(2), ammo(1), reload(1)]
        
        In the ego frame, forward direction = [1, 0] (the ego ship's heading is
        rotated to zero). This means:
          - cos_bearing = unit_to_opp[0]:  1.0 = dead ahead, -1.0 = behind
          - sin_bearing = unit_to_opp[1]:  sign directly indicates turn direction
        
        Added features:
          - dist:            scalar distance (avoids network learning sqrt)
          - closing_speed:   positive = gap is shrinking (requires dot product)
          - cos_bearing:     aim quality — how close to crosshairs
          - sin_bearing:     aim correction direction — which way to turn
          - opp_facing_me:   threat level — is opponent's nose pointed at us
          - cos_lead_error:  deflection aim quality — accounts for relative motion
                             1.0 = firing now would lead the target correctly
          - sin_lead_error:  deflection aim correction direction
        """
        opp_pos = opp_raw[0:2]
        opp_vel = opp_raw[2:4]
        opp_auv = opp_raw[4:6]

        dist = np.linalg.norm(opp_pos)

        if dist > 1e-8:
            opp_dir = opp_pos / dist
            rel_vel = opp_vel - vel_ego
            closing_speed = -np.dot(rel_vel, opp_dir)
            cos_bearing = opp_dir[0]
            sin_bearing = opp_dir[1]
            opp_facing_me = -np.dot(opp_auv, opp_dir)

            # --- Lead angle (first-order intercept approximation) ---
            # Estimate: time for a missile to cover current distance
            #   missile_vel_ego_frame = vel_ego + MISSILE_VEL * [1, 0]
            #   Relative missile speed toward target ≈ MISSILE_VEL (dominant term)
            # Approximate flight time:
            t_flight = dist / (MISSILE_VEL + 1e-8)
            # Where the opponent will be (linear extrapolation):
            lead_pos = opp_pos + rel_vel * t_flight
            lead_dist = np.linalg.norm(lead_pos)
            if lead_dist > 1e-8:
                lead_dir = lead_pos / lead_dist
                cos_lead_error = lead_dir[0]
                sin_lead_error = lead_dir[1]
            else:
                cos_lead_error = 1.0
                sin_lead_error = 0.0
        else:
            closing_speed = 0.0
            cos_bearing = 1.0
            sin_bearing = 0.0
            opp_facing_me = 0.0
            cos_lead_error = 1.0
            sin_lead_error = 0.0

        derived = np.array([
            dist / WRAP_BOUND,        # ~[0, 1.4]
            closing_speed,            # ~[-2.8, 2.8]
            cos_bearing,              # [-1, 1]
            sin_bearing,              # [-1, 1]
            opp_facing_me,            # [-1, 1]
            cos_lead_error,           # [-1, 1]
            sin_lead_error,           # [-1, 1]
        ], dtype=np.float32)

        return np.concatenate([opp_raw, derived])

    def _augment_missile_obs(self, missile, vel_ego):
        """
        Derive threat/tracking features for a single projectile.
        
        Raw layout (egocentric): [rel_pos(2), rel_vel(2), life_frac(1)]
        
        Uses linear closest-approach calculation:
          Given relative position p and relative velocity v,
          time of closest approach = -dot(p, v) / dot(v, v)
          miss distance = |p + v * t_closest|
        
        This is approximate (ignores gravity curving the trajectory), but over
        typical missile flight times the error is small, and the network can
        learn to compensate using the orbital context features on self_obs.
        
        Added features:
          - distance:        scalar distance to missile
          - t_closest_norm:  when the missile will be closest (0=now, 1=far future)
                             clamped to [0, 1] via missile lifetime
          - miss_distance:   predicted closest approach distance
                             low value + low t_closest = EVADE NOW
        """
        m_raw = missile.get_obs(self.playerShips[0])
        m_pos = m_raw[0:2]
        m_vel = m_raw[2:4]

        dist = np.linalg.norm(m_pos)

        # Velocity of missile relative to ego ship
        rel_vel = m_vel - vel_ego
        rel_speed_sq = np.dot(rel_vel, rel_vel)

        if rel_speed_sq > 1e-12:
            # Time of closest approach (linear model)
            t_closest = -np.dot(m_pos, rel_vel) / rel_speed_sq
            t_closest = max(0.0, t_closest)  # Only future matters
            # Position at closest approach
            closest_pos = m_pos + rel_vel * t_closest
            miss_dist = np.linalg.norm(closest_pos)
        else:
            # Missile co-moving with us or stationary relative to us
            t_closest = 0.0
            miss_dist = dist

        # Normalize time by missile lifetime for bounded feature
        t_norm = min(t_closest / max(MISSILE_LIFE, 1.0), 1.0)

        derived = np.array([
            dist / WRAP_BOUND,          # ~[0, 1.4]
            t_norm,                     # [0, 1]
            miss_dist / WRAP_BOUND,     # ~[0, 1.4]
        ], dtype=np.float32)

        return np.concatenate([m_raw, derived])

    def get_obs(self):
        ego = self.playerShips[0] if self.egocentric else None

        self_raw = self.playerShips[0].get_obs(ego)
        opp_raw = self.playerShips[1].get_obs(ego)

        if ego is not None:
            # Ego velocity in ego frame (needed by all augmentation methods)
            vel_ego = self_raw[2:4]

            self_obs = self._augment_self_obs(self_raw, vel_ego)
            opp_obs = self._augment_opp_obs(opp_raw, vel_ego)

            friendly_missiles = [
                self._augment_missile_obs(m, vel_ego)
                for m in self.missiles
            ]
            hostile_missiles = [
                self._augment_missile_obs(m, vel_ego)
                for m in self.opponent_missiles
            ]
        else:
            # Non-egocentric: no augmentation (features assume ego frame)
            self_obs = self_raw
            opp_obs = opp_raw
            friendly_missiles = [m.get_obs(None) for m in self.missiles]
            hostile_missiles = [m.get_obs(None) for m in self.opponent_missiles]

        obs = {
            "self": self_obs,
            "opponent": opp_obs,
            "missiles_friendly": self.missile_space.encode_obs(friendly_missiles),
            "missiles_hostile": self.missile_space.encode_obs(hostile_missiles),
        }
        return {0: obs}

    def get_keymap(self):
        return {0: {pygame.K_UP: (0, 1, False),
                     pygame.K_LEFT: (1, 1, False), pygame.K_RIGHT: (1, 2, False),
                     pygame.K_DOWN: (2, 1, False),
                     "default": 0},
                }

    def new_target_position(self):
        # ... (unchanged)
        target = self.playerShips[1]
        if (self.target_speed == 0):
            position = np.random.uniform(-WRAP_BOUND, WRAP_BOUND, (2,))
        else:
            position = np.random.uniform(-WRAP_BOUND, WRAP_BOUND, (2,))
            r = np.random.uniform(0, 1) ** .5 * WRAP_BOUND
            p_ang = np.random.uniform(0, 2 * np.pi)
            position = np.array([np.cos(p_ang), np.sin(p_ang)]) * r
            dist_zero = (position ** 2).sum() ** .5
            player_pos = self.playerShips[0].pos
            disp = (position - player_pos)
            bounds_dist = MISSILE_LIFE * MISSILE_VEL * 1.2
            dist_from_valid = bounds_dist - (disp ** 2).sum() ** .5
            if (dist_from_valid > 0):
                position = player_pos + disp / (1 - dist_from_valid / bounds_dist)
                pos_r = (position ** 2).sum() ** .5
                if (pos_r > WRAP_BOUND):
                    position = position * -(2 * WRAP_BOUND - pos_r) / pos_r
            g = GRAV_CONST
            if (not self.elliptical):
                v_magnitude = (g / r) ** .5
            else:
                r_p = np.random.uniform(low=max(PLAYER_SIZE * 2 + STAR_SIZE, r / 2), high=r)
                ecc = (r - r_p) / (r + r_p)
                v_magnitude = ((1 - ecc) * g / ((1 + ecc) * r)) ** .5
            v_magnitude *= self.target_speed
            v_angle = np.arctan2(position[1], position[0]) + np.pi / 2 * np.sign(np.random.rand() - 0.5)
            target.vel = np.array([np.cos(v_angle), np.sin(v_angle)]) * v_magnitude
        target.pos = position
        if (self.target_ammo != 0):
            target.stored_missiles = int(NUM_MISSILES * self.target_ammo)
            target.ang = np.random.uniform(0, 360)
            target.updateAngUV()
        else:
            target.stored_missiles = 0

    def reset(self, seed=None, options={}):
        self.playerShips = [
            Ship(np.array([-.5, .5]), 90.),
            Dummy_Ship(np.array([0., 0.]), 0., PLAYER_SIZE * self.size_multiplier)
        ]
        self.new_target_position()
        self.missiles = []
        self.opponent_missiles = []
        self.time = 0
        self.terminated = False
        return self.get_obs(), {}

    def shaped_reward(self, ship, target, base_reward):
        r = base_reward
        r += 0.0002
        dist_to_star = np.linalg.norm(ship.pos)
        if dist_to_star < STAR_SIZE * 3:
            r -= 0.005 * (1 - dist_to_star / (STAR_SIZE * 3))
        for m in self.missiles:
            dist = np.linalg.norm(m.pos - target.pos)
            r += 0.0005 / (dist + 0.1)
        for m in self.opponent_missiles:
            dist = np.linalg.norm(m.pos - ship.pos)
            if dist < 0.2:
                r -= 0.002 * (1 - dist / 0.2)
        return r

    def step(self, actions):
        self.rewards = {0: 0}
        self.time += 1 * self.speed
        ship = self.playerShips[0]
        ship.update(actions[0], self.missiles, self.speed, grav_multiplier=self.grav_multiplier)
        if (np.linalg.norm(ship.pos, 2) < PLAYER_SIZE + STAR_SIZE):
            self.terminated = True
            self.rewards[0] = -1
        target = self.playerShips[1]
        if ((self.target_speed != 0) or (target.stored_missiles != 0)):
            target.update(self.opponent_missiles, self.speed,
                          grav_multiplier=self.grav_multiplier * self.target_speed, target_loc=ship.pos)
            if (np.linalg.norm(target.pos, 2) < PLAYER_SIZE + STAR_SIZE):
                self.new_target_position()
        ms = [self.missiles, self.opponent_missiles]
        for m in ms:
            for i in reversed(range(len(m))):
                si, d = m[i].update(self.playerShips, self.speed)
                if (d):
                    del m[i]
                if (si != -1):
                    if (si == 0):
                        self.terminated = True
                        self.rewards[0] = -1
                    else:
                        self.rewards[0] = 1
                        self.new_target_position()
        truncated = (self.time >= self.maxTime)
        info = {BASE_REWARD: self.rewards[0]}
        return self.get_obs(), self.rewards, {"__all__": self.terminated}, {"__all__": truncated}, info

    def render(self):
        # ... (unchanged from original)
        ego = self.playerShips[0] if self.render_egocentric else None
        dim = self.size
        hdim = dim / 2
        ssz = STAR_SIZE * dim
        msz = 3
        img = Image.new('RGB', (dim, dim), color='black')
        draw = ImageDraw.Draw(img)
        if (ego is None):
            for i in range(2):
                ss = np.random.uniform(-1.0, 1.0, 2) * ssz / 2
                draw.line((hdim + ss[0], hdim + ss[1], hdim - ss[0], hdim - ss[1]), fill='white', width=1)
            rs = (1 - WRAP_BOUND) * hdim
            draw.rectangle((rs, rs, dim - rs, dim - rs), outline='white')
        ship = self.playerShips[0]
        ship.render(draw, dim, hdim, ssz, self.terminated, ego=ego)
        target = self.playerShips[1]
        target.render(draw, hdim, ego)
        for m in self.missiles:
            m.render(draw, hdim, msz, self.speed, ego=ego)
        for m in self.opponent_missiles:
            m.render(draw, hdim, msz, self.speed, ego=ego, c="orange")
        if (self.egocentric):
            img = img.rotate(90)
            draw = ImageDraw.Draw(img)
        draw.text((10, img.height - 60), f"Time: {self.time}/{self.maxTime}", (255, 255, 255))
        return img