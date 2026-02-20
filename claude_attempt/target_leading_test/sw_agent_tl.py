import torch
import torch.nn as nn
import numpy as np
from gymnasium.spaces import Dict, Box, MultiDiscrete

from sw_env import missile_space

class SpaceWarNet(nn.Module):
    def __init__(self, ship_size=8, missile_size=5, max_missiles=10, hidden=256):
        super().__init__()
        
        # Encode each ship independently
        self.self_encoder = nn.Sequential(
            nn.Linear(ship_size, 64),
        )
        self.opp_encoder = nn.Sequential(
            nn.Linear(ship_size, 64),
        )
        
        self.missile_encoder = nn.Sequential(
            nn.Linear(missile_size, 64),
        )
        
        # Trunk after combining all features
        # 2 ships * 64 + 2 missile sets * 64 = 256
        self.trunk = nn.Sequential(
            nn.Linear(192, hidden),#nn.Linear(256, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
        )
        
        # Separate policy heads for each action dimension
        self.policy_turn = nn.Linear(hidden, 2)      # nop/left/right
        self.policy_shoot = nn.Linear(hidden, 2)     # nop/shoot
        
        # Value head
        self.value_head = nn.Linear(hidden, 1)
        
    def forward(self, obs_dict):
        self_obs = obs_dict["self"]           # (batch, 8)
        opp_obs = obs_dict["opponent"]        # (batch, 8)
        friendly_m = obs_dict["missiles_friendly"]  # (batch, max_m * 5)
        
        if (len(self_obs.shape)==1):
            self_obs = self_obs.unsqueeze(0)
            opp_obs = opp_obs.unsqueeze(0)
            friendly_m = friendly_m.unsqueeze(0)
        
        self_enc = self.self_encoder(self_obs)
        opp_enc = self.opp_encoder(opp_obs)
        friendly_enc = self.missile_encoder(friendly_m)
        combined = torch.cat([self_enc, opp_enc, friendly_enc], dim=-1) #, hostile_enc], dim=-1)
        features = self.trunk(combined)
        
        logits_turn = self.policy_turn(features)
        logits_shoot = self.policy_shoot(features)
        
        value = self.value_head(features)
        return [logits_turn, logits_shoot], value

# Use attention instead of a sequential trunk
class SimpleTransformerLayer(nn.Module): # A simplified transformer layer
    def __init__(self, emb_dim, heads, h_dim=2048, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.mha = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        self.norm_attn = torch.nn.LayerNorm(emb_dim)
        self.norm_ff = torch.nn.LayerNorm(emb_dim)
        self.residual = nn.Sequential(
            nn.Linear(emb_dim, h_dim),
            nn.GELU(), # Apparently just plain better than ReLU here.
            nn.Dropout(dropout),
            nn.Linear(h_dim, emb_dim),
            nn.Dropout(dropout),
        )
    def forward(self, x, src_key_padding_mask):
        x_attn, _ = self.mha(x, x, x, key_padding_mask=src_key_padding_mask, need_weights=False)
        x = self.norm_attn(x_attn + x)
        x_ff = self.residual(x)
        x = self.norm_ff(x_ff + x)
        return x

class SpaceWarNet_Attention(nn.Module):
    def __init__(self, ship_size=8, missile_size=5, max_missiles=10, hidden=256, emb=64):
        super().__init__()
        
        # Encode each ship independently
        self.self_encoder = nn.Sequential(
            nn.Linear(ship_size, emb),
        )
        self.opp_encoder = nn.Sequential(
            nn.Linear(ship_size, emb),
        )
        
        self.missile_encoder = nn.Sequential(
            nn.Linear(missile_size, emb),
        )
        
        # In the final version, maybe we run attention on all objects, then max-pool over object classes.
        # [player]+[target]+[missile,missile,missile]->attention->pool->[player+target+pooled_missiles]->FF
        #self.object_attention = nn.MultiheadAttention(embed_dim=emb, num_heads=4, batch_first=True)
        self.object_attention = SimpleTransformerLayer(emb, heads=4, h_dim=hidden, dropout=0.0)
        
        # Trunk after combining all features
        # 2 ships * 64 + 2 missile sets * 64 = 256
        self.trunk = nn.Sequential(
            nn.Linear(emb*3, hidden),#nn.Linear(256, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
        )
        
        # Separate policy heads for each action dimension
        self.policy_turn = nn.Linear(hidden, 2)      # nop/left/right
        self.policy_shoot = nn.Linear(hidden, 2)     # nop/shoot
        
        # Value head
        self.value_head = nn.Linear(hidden, 1)
        
    def forward(self, obs_dict):
        self_obs = obs_dict["self"]           # (batch, 8)
        opp_obs = obs_dict["opponent"]        # (batch, 8)
        friendly_m = obs_dict["missiles_friendly"]  # (batch, 5)
        
        if (len(self_obs.shape)==1):
            self_obs = self_obs.unsqueeze(0)
            opp_obs = opp_obs.unsqueeze(0)
            friendly_m = friendly_m.unsqueeze(0)
        
        self_enc = self.self_encoder(self_obs)
        opp_enc = self.opp_encoder(opp_obs)
        friendly_enc = self.missile_encoder(friendly_m)
        batch_size = self_enc.shape[0]
        
        combined = torch.stack([self_enc, opp_enc, friendly_enc], dim=1)
        features = self.object_attention(combined,src_key_padding_mask=None) # Self-attention
        features = features.reshape((batch_size, -1))
        features = self.trunk(features)
        
        logits_turn = self.policy_turn(features)
        logits_shoot = self.policy_shoot(features)
        
        value = self.value_head(features)
        return [logits_turn, logits_shoot], value