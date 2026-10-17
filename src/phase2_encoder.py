# ============================================================
# Phase 2 — DINOv2 feature extraction & frozen projector P_phi
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class Projector(nn.Module):
    def __init__(self, in_dim=768, out_dim=128, hidden=256, depth=2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        layers += [nn.LayerNorm(out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GatedContactPredictor(nn.Module):
    def __init__(self, K=128, action_dim=2, hidden=256, depth=2):
        super().__init__()
        def block():
            layers = []
            d = K + action_dim
            for _ in range(depth):
                layers += [nn.Linear(d, hidden), nn.GELU()]
                d = hidden
            layers += [nn.Linear(d, K)]
            return nn.Sequential(*layers)

        self.f_fs = block()
        self.f_ct = block()
        self.gate = nn.Linear(K, 2)

        for head in (self.f_fs, self.f_ct):
            last = head[-1]
            nn.init.normal_(last.weight, std=1e-2)
            nn.init.zeros_(last.bias)

    def forward(self, z, a):
        if a.dim() == 1:
            a = a.unsqueeze(0).expand(z.size(0), -1)
        rho = F.softmax(self.gate(z), dim=-1)
        za = torch.cat([z, a], dim=-1)
        return z + rho[:, 0:1] * self.f_fs(za) + rho[:, 1:2] * self.f_ct(za)
