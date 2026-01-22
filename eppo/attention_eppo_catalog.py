import gymnasium as gym

from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.configs import ActorCriticEncoderConfig

from classes.attention_encoder import AttentionPPOCatalog

class AttentionEPPOCatalog(AttentionPPOCatalog):
    @override(PPOCatalog)
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,  # TODO: Remove?
        model_config_dict: dict,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
        )
        self.vf_head_config.output_layer_dim = 4 # gamma, v, alpha, beta