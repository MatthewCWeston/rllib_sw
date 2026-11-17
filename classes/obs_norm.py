import torch
from torch import nn

def _merge(a, b): # Merge two sets of samples
    (n_a, m_a, s_a), (n_b, m_b, s_b) = a, b
    n = n_a + n_b
    d = m_b - m_a
    return n, m_a + d * n_b / n.clamp(min=1), s_a + s_b + d ** 2 * n_a * n_b / n.clamp(min=1)


class ObsNorm(nn.Module):
    def __init__(self, embs, clip=10.0, eps=1e-8):
        super().__init__()
        self.clip, self.eps, self.pending, self.embs = clip, eps, {}, dict(embs)
        # Each entity class with Linear embedding layers gets its own normalization here.
        for k, d in ((k, e.in_features) for k, e in embs.items()):
            self.register_buffer(f"{k}_count", torch.zeros(d))
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
        for key, (n_new, u_new, var_new) in self.pending.items(): # Count, mean, variance
            count, mean, var = (getattr(self, f"{key}_{s}") for s in ("count", "mean", "var"))
            active = (count > 0) | (var_new > 1e-12 * n_new) # Account for features that exist in early stages but only vary in later ones. 'active' checks that a feature changes within a batch.
            n, m, s = _merge((count.double(), mean.double(), var.double() * count.double()), (n_new * active, u_new, var_new))
            count[active], mean[active], var[active] = n[active].float(), m[active].float(), (s[active] / n[active]).float
            emb = self.embs[key] # Zero weights until active; fold weights into bias
            emb.bias += emb.weight[:, ~active] @ u_new[~active].to(emb.weight.dtype)
            emb.weight[:, ~active] = 0
        self.pending = {}