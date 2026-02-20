import torch
import torch.nn as nn
import numpy as np
from gymnasium.spaces import Dict, Box, MultiDiscrete

from sw_env import missile_space

class SpaceWarNet(nn.Module):
    def __init__(self, ship_size=8, missile_size=5, max_missiles=10, hidden=256):
        super().__init__()
        
        # Encode self and opponent ships
        self.opp_encoder = nn.Sequential(
            nn.Linear(ship_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
        )
        self.self_encoder = nn.Sequential(
            nn.Linear(ship_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
        )
        
        # Encode missiles with permutation-invariant aggregation
        # Each missile set gets its own encoder, then we pool
        self.missile_encoder = nn.Sequential(
            nn.Linear(missile_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
        )
        
        # Trunk after combining all features
        # 2 ships * 64 + 2 missile sets * 64 = 256
        self.trunk = nn.Sequential(
            nn.Linear(256, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
        )
        
        # Separate policy heads for each action dimension
        self.policy_thrust = nn.Linear(hidden, 2)   # nop/thrust
        self.policy_turn = nn.Linear(hidden, 3)      # nop/left/right
        self.policy_shoot = nn.Linear(hidden, 2)     # nop/shoot
        
        # Value head
        self.value_head = nn.Linear(hidden, 1)
        
    def encode_missiles(self, missile_obs):
        """
        missile_obs: (batch, max_missiles * missile_size) from RepeatedCustom encoding
        Returns: (batch, 64) - pooled missile features
        """
        batch_size = missile_obs.shape[0]
        missile_size = missile_obs.shape[-1]
        # MCW: Used RepeatedCustom to encode this space; decode first
        v, mask = missile_space.decode_obs(missile_obs) # Returns a [batch x max_seq_len x feature_size] array and a [batch x max_seq_len] mask
        max_len = int(mask.sum(dim=1).max().item())
        if (max_len==0):
            return torch.zeros((batch_size, 64))
        v = v[:,:max_len,:] # Improve efficiency for 'sparse' repeated spaces.
        mask = mask[:,:max_len]
        
        # Encode each missile
        encoded = self.missile_encoder(v)  # (batch, max_missiles, 64)
        
        # Masked mean pooling
        mask_expanded = mask.unsqueeze(-1)  # (batch, max_missiles, 1)
        pooled = (encoded * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        return pooled
    
    def forward(self, obs_dict):
        self_obs = obs_dict["self"]           # (batch, 8)
        opp_obs = obs_dict["opponent"]        # (batch, 8)
        friendly_m = obs_dict["missiles_friendly"]  # (batch, max_m * 5)
        hostile_m = obs_dict["missiles_hostile"]     # (batch, max_m * 5)
        
        if (len(self_obs.shape)==1):
            self_obs = self_obs.unsqueeze(0)
            opp_obs = opp_obs.unsqueeze(0)
            friendly_m = friendly_m.unsqueeze(0)
            hostile_m = hostile_m.unsqueeze(0)
        
        self_enc = self.self_encoder(self_obs)
        opp_enc = self.opp_encoder(opp_obs)
        friendly_enc = self.encode_missiles(friendly_m)
        hostile_enc = self.encode_missiles(hostile_m)
        combined = torch.cat([self_enc, opp_enc, friendly_enc, hostile_enc], dim=-1)
        features = self.trunk(combined)
        
        logits_thrust = self.policy_thrust(features)
        logits_turn = self.policy_turn(features)
        logits_shoot = self.policy_shoot(features)
        
        value = self.value_head(features)
        
        return [logits_thrust, logits_turn, logits_shoot], value