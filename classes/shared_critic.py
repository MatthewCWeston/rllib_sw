from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule

from torch import nn

SHARED_CRITIC_ID = "shared_critic"
OPPONENT_OBS = 'opponent_obs'

class SharedCriticModule(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        self.encoder = nn.Module()
        self.encoder.critic_encoder = self.catalog._encoder_config.build(framework=self.framework, is_critic=True)
        self.vf = self.catalog.build_vf_head(framework=self.framework)

    def compute_values(self, batch, embeddings=None):
        if self.model_config.get("vf_symmetry"):
            obs = batch[Columns.OBS]
            f = lambda o: self.vf(self.encoder.critic_encoder({Columns.OBS: o})[ENCODER_OUT]).squeeze(-1)
            return (f({k: v for k, v in obs.items() if k != OPPONENT_OBS}) - f(obs[OPPONENT_OBS])) / 2 # Antisymmetric part of f
        if embeddings is None:
            embeddings = self.encoder.critic_encoder(batch)[ENCODER_OUT]
        return self.vf(embeddings).squeeze(-1)