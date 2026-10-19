import torch
from torch import nn

def _merge(a, b): # Merge two sets of samples
    (n_a, m_a, s_a), (n_b, m_b, s_b) = a, b
    n = n_a + n_b
    d = m_b - m_a
    return n, m_a + d * n_b / n.clamp(min=1), s_a + s_b + d ** 2 * n_a * n_b / n.clamp(min=1)


class ObsNorm(nn.Module):
    def __init__(self, obs_space, clip=10.0, eps=1e-8):
        super().__init__()
        self.clip, self.eps, self.pending = clip, eps, {}
        # Each entity class with Linear embedding layers gets its own normalization here.
        for k, d in obs_space.items():
            self.register_buffer(f"{k}_count", torch.zeros(()))
            self.register_buffer(f"{k}_mean", torch.zeros(d))
            self.register_buffer(f"{k}_var", torch.ones(d))

    def forward(self, key, v, mask):  # v: [batch, tokens, features], mask: [batch, tokens]
        if not hasattr(self, f"{key}_mean"):  # Only normalize Box observations
            return v
        if self.training: # Only update weights on iteration completion
            with torch.no_grad():
                w, x = mask.unsqueeze(-1).double(), v.detach().double()
                n = w.sum()
                mean = (w * x).sum((0, 1)) / n.clamp(min=1)
                new = (n, mean, (w * (x - mean) ** 2).sum((0, 1)))
                self.pending[key] = _merge(self.pending[key], new) if key in self.pending else new
        mean, var = getattr(self, f"{key}_mean"), getattr(self, f"{key}_var")
        return ((v - mean) / torch.sqrt(var + self.eps)).clamp(-self.clip, self.clip)

    @torch.no_grad()
    def commit(self):
        for key, new in self.pending.items():
            count, mean, var = (getattr(self, f"{key}_{s}") for s in ("count", "mean", "var"))
            n, m, s = _merge((count.double(), mean.double(), var.double() * count.double()), new)
            if n > 0:
                count.copy_(n), mean.copy_(m), var.copy_(s / n)
        self.pending = {}