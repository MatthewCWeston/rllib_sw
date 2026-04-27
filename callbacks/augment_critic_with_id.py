from typing import Any, Optional
from collections.abc import Callable

import gymnasium as gym
import numpy as np

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.env_to_module.observation_preprocessor import (
    MultiAgentObservationPreprocessor,
)
from ray.rllib.utils.annotations import override

import inspect

from classes.attention_encoder import CRITIC_ONLY

OPPONENT_ID = f'{CRITIC_ONLY}_opponent_id'

class AugmentCriticWithOpponentID(MultiAgentObservationPreprocessor):

    @override(ConnectorV2)
    def __init__(
        self,
        module_name_to_id: Callable,
        max_opponents: int,
        input_observation_space: Optional[gym.Space] = None,
        input_action_space: Optional[gym.Space] = None,
        **kwargs,
    ):
        super().__init__(input_observation_space, input_action_space, **kwargs,)
        self.module_name_to_id = module_name_to_id
        self.max_opponents = max_opponents

    @override(MultiAgentObservationPreprocessor)
    def recompute_output_observation_space(
        self,
        input_observation_space,
        input_action_space,
    ) -> gym.Space:
        spaces = {}
        for k, v in input_observation_space.items():
          new_agent_obs = {k2: v2 for k2, v2 in v.items()}
          new_agent_obs[OPPONENT_ID] = gym.spaces.Discrete(self.max_opponents)
          spaces[k] = gym.spaces.Dict(new_agent_obs)
        return gym.spaces.Dict(spaces)

    @override(MultiAgentObservationPreprocessor)
    def preprocess(self, observations, episode) -> Any:
        if (observations == {}):
          return observations # Empty observations (e.g. on episode complete)
        assert len(observations) == 2 # This is for 2 agent MARL environments.
        agents = list(observations.keys())
        opponents = {a1: a2 for a1, a2 in zip(agents, agents[::-1])}
        new_obs = {}
        for k, v in observations.items():
          new_obs[k] = {k2: v2 for k2, v2 in v.items()}
          new_obs[k][OPPONENT_ID] = self.module_name_to_id(episode.module_for(opponents[k]))
        return new_obs