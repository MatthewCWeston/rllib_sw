import typing
from typing import Any, Optional

from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
from ray.util.annotations import DeveloperAPI

from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule
)

from eppo.attention_eppo_catalog import AttentionEPPOCatalog

torch, nn = try_import_torch()


@DeveloperAPI
class EPPOTorchRLModule(DefaultPPOTorchRLModule):
    
    @override(DefaultPPOTorchRLModule)
    def __init__(self, *args, **kwargs):
        catalog_class = kwargs.pop("catalog_class", None)
        if catalog_class is None:
            catalog_class = AttentionEPPOCatalog
        # Skip DefaultPPOTorchRLModule.__init__
        super(DefaultPPOTorchRLModule, self).__init__(*args, **kwargs, catalog_class=catalog_class)

    @override(DefaultPPOTorchRLModule)
    def compute_values(
        self,
        batch: typing.Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        # Used in the general case, and by GAE in particular
        if embeddings is None:
            if hasattr(self.encoder, "critic_encoder"):
                embeddings = self.encoder.critic_encoder(batch)[ENCODER_OUT]
            else:
                embeddings = self.encoder(batch)[ENCODER_OUT][CRITIC]
        vf_out = self.vf(embeddings) # 
        gamma, _,_,_ = [x.squeeze() for x in vf_out.chunk(4, dim=-1)]
        return gamma
        
    def compute_value_distributions(
        self,
        batch: typing.Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        # Used by the learner, to optimize the predicted value distribution
        if embeddings is None:
            embeddings = self.encoder(batch)[ENCODER_OUT]
        vf_out = self.vf(embeddings)
        gamma, logv, log_alpha_minus_one, log_beta = [x.squeeze() for x in vf_out.chunk(4, dim=-1)]
        return gamma, torch.exp(logv), torch.exp(log_alpha_minus_one) + 1, torch.exp(log_beta)