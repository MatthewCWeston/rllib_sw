import numpy as np

# Constants
WRAP_BOUND = .5**.5 # x/y distance from star at which wrapping occurs 

NUM_MISSILES = 32
MISSILE_VEL = 1/256.
MISSILE_LIFE = 96
MISSILE_RELOAD_TIME = 16

SHIP_FUEL = 1024 # How many timesteps can the ship thrust for?

SHIP_TURN_RATE = 1/16 * 180/np.pi   # About 3.6 degrees
SHIP_ACC = 1/131072 * WRAP_BOUND * 2    # About 5E-6
GRAV_CONST = 9.5E-7 * .5**.5 * 2              # PENDING

DEFAULT_MAX_TIME = 4096

DEFAULT_RENDER_SIZE = 750
PLAYER_SIZE = .02 * WRAP_BOUND * 2 # Player size
STAR_SIZE = .01 * WRAP_BOUND * 2 # Star size

HYPERSPACE_CHARGES = 8 # How many times can the ship enter hyperspace?
HYPERSPACE_RECHARGE = 224 # Time for hyperspace to recharge after activation (12 seconds)
HYPERSPACE_REENTRY = 96 # Time after re-entering (5 seconds)

S_HSPACE_MAXSPEED = 1/131072 * 256 * WRAP_BOUND * 2 # Maximum speed when leaving hyperspace, when stochastic hyperspace is enabled