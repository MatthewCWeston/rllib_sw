# Constants
NUM_MISSILES = 32
MISSILE_VEL = 0.005
MISSILE_LIFE = 108
MISSILE_RELOAD_TIME = 18

SHIP_FUEL = 1152 # How many timesteps can the ship thrust for?

SHIP_TURN_RATE = 1.5
SHIP_ACC = .000005
GRAV_CONST = .00000125

DEFAULT_MAX_TIME = 1024

DEFAULT_RENDER_SIZE = 750
PLAYER_SIZE = .02 # Player size
STAR_SIZE = .01 # Star size

WRAP_BOUND = .5**.5 # x/y distance from star at which wrapping occurs 

HYPERSPACE_CHARGES = 8 # How many times can the ship enter hyperspace?
HYPERSPACE_RECHARGE = 230 # Time for hyperspace to recharge after activation (12 seconds)
HYPERSPACE_REENTRY = 96 # Time after re-entering (5 seconds)

S_HSPACE_MAXSPEED = 0.005 # Maximum speed when leaving hyperspace, when stochastic hyperspace is enabled
S_HSPACE_FAIL_CHANCE = 1/8 # Odds of spontaneous explosion when leaving hyperspace, when stochastic hyperspace is enabled