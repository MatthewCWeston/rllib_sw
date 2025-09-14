"""
This file implements a self-attention Encoder (see https://arxiv.org/abs/1909.07528) for handling variable-length Repeated observation spaces. It expects a Dict observation space with Discrete, Box, or Repeated (of Discrete or Box) subspaces.
"""
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.base import Encoder, ENCODER_OUT
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.columns import Columns

import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from classes.repeated_space import RepeatedCustom

import torch
from torch import nn
import torch.nn.functional as F

class SimpleTransformerLayer(nn.Module): # A simplified transformer layer
    '''
        https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/transformer.py#L933
        Official implementation also uses self-attn, but includes a (default) 2048 dimension hidden layer (with ReLU activation), layer normalization, and some dropout layers.
        1. x = x + dropout(self_attn(x))
        2. x = Layernorm_1(x)
        3. x = x + ff_module(x)
        4. x = Layernorm_2(x)
        
        Layernorm subtracts mean and divides by standard deviation. The dropout layers are vital to regularization, at least for emb_dim=128, and the network falls apart without them.
    '''
    def __init__(self, emb_dim, heads, h_dim=2048, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.mha = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        self.norm_attn = torch.nn.LayerNorm(emb_dim)
        self.norm_ff = torch.nn.LayerNorm(emb_dim)
        self.residual = nn.Sequential(
            nn.Linear(emb_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, emb_dim),
            nn.Dropout(dropout),
        )
    def forward(self, x, src_key_padding_mask):
        x_attn, _ = self.mha(x, x, x, key_padding_mask=src_key_padding_mask, need_weights=False)
        x_attn = F.dropout(x_attn, self.dropout)
        x = self.norm_attn(x_attn + x)
        x_ff = self.residual(x)
        x = self.norm_ff(x_ff + x)
        return x

class AttentionEncoder(TorchModel, Encoder):
    """
    An Encoder that takes a Dict of multiple spaces, including Discrete, Box, and Repeated, and uses an attention layer to convert this variable-length input into a fixed-length featurized learned representation.
    """

    def __init__(self, config):
        try:
            super().__init__(config)
            self.observation_space = config.observation_space
            self.emb_dim = config.emb_dim
            self.recursive = config.recursive
            self.attn_layers = config.attn_layers
            # Use an attention layer to reduce observations to a fixed length
            mhas = []
            for _ in range(self.attn_layers):
                if (config.full_transformer):
                    mhas.append(nn.TransformerEncoderLayer(d_model=self.emb_dim, 
                        dim_feedforward=config.attn_ff_dim, nhead=4, batch_first=True))
                else:
                    mhas.append(SimpleTransformerLayer(self.emb_dim, 4,
                        h_dim=config.attn_ff_dim))
                if (self.recursive): # If recursive, only create one layer
                    break
            self.mha = nn.ModuleList(mhas)
            # Can just run a bunch of these in sequence, they are self-contained.
            # Set up embedding layers for each element in our observation
            embs = {}
            for n, s in self.observation_space.spaces.items():
                if type(s) is RepeatedCustom:
                    s = s.child_space  # embed layer applies to child space
                if type(s) is Box:
                    embs[n] = nn.Linear(s.shape[0], self.emb_dim)
                elif type(s) is Discrete:
                    embs[n] = nn.Embedding(s.n, self.emb_dim)
                else:
                    raise Exception("Unsupported observation subspace")
            self.embs = nn.ModuleDict(embs)
        except Exception as e:
            print("Exception when building AttentionEncoder:")
            print(e)
            raise e

    def _forward(self, input_dict, **kwargs):
        obs = input_dict[Columns.OBS]
        # The original space we mapped from.
        obs_s = self.observation_space
        embeddings = []
        masks = []
        for s in obs.keys():
            v = obs[s]
            v_s = obs_s[s]
            if type(v_s) is RepeatedCustom:
                v, mask = v_s.decode_obs(v)
                max_len = int(mask.sum(dim=1).max().item())
                if (max_len==0):    # Skip repeated spaces with no items
                    continue
                v = v[:,:max_len,:] # Improve efficiency for 'sparse' repeated spaces.
                mask = mask[:,:max_len]
            elif type(v_s) in [Box, Discrete]:
                mask = torch.ones((v.shape[0], 1)).to(
                    v.device
                ) # Fixed elements are always there
                v = v.unsqueeze(1) # Add sequence length dimension after batch
            embedded = self.embs[s](v)
            embeddings.append(embedded)
            masks.append(mask)
        # All entities have embeddings. Apply masked residual self-attention and then mean-pool.
        x = torch.concatenate(embeddings, dim=1)  # batch_size, seq_len, unit_size
        mask = torch.concatenate(masks, dim=1)  # batch_size, seq_len
        for i in range(self.attn_layers):
            layer = self.mha[0] if self.recursive else self.mha[i]
            x = layer(x, src_key_padding_mask=mask)
        # Masked mean-pooling.
        mask = mask.unsqueeze(dim=2)
        x = x * mask  # Mask x to exclude nonexistent entries from mean pool op
        x = x.mean(dim=1) * mask.shape[1] / mask.sum(dim=1)  # Adjust mean
        return {ENCODER_OUT: x}


class AttentionEncoderConfig(ModelConfig):
    """
    Produces an AttentionEncoder.

    kwargs:
     * attention_emb_dim: The embedding dimension of the attention layer.
    """

    def __init__(self, observation_space, **kwargs):
        self.observation_space = observation_space
        self.emb_dim = kwargs["model_config_dict"]["attention_emb_dim"]
        self.attn_ff_dim = kwargs["model_config_dict"]["attn_ff_dim"]
        self.full_transformer = kwargs["model_config_dict"]["full_transformer"]
        self.attn_layers = kwargs["model_config_dict"]["attn_layers"]
        self.recursive = kwargs["model_config_dict"]["recursive"]
        self.output_dims = (self.emb_dim,)

    def build(self, framework):
        return AttentionEncoder(self)

    def output_dims(self):
        return self.output_dims

class AttentionPPOCatalog(PPOCatalog):
    """
    A special PPO catalog producing an encoder that handles dictionaries of (potentially Repeated) action spaces in the same manner as https://arxiv.org/abs/1909.07528.
    """

    @classmethod
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        **kwargs,
    ):
        return AttentionEncoderConfig(observation_space, **kwargs)