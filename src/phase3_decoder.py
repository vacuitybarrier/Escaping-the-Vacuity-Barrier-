# ============================================================
# Phase 3 — Decoder transfer evaluation and P1b gate diagnostics
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseDecoder(nn.Module):
    """Linear probe and non-linear decoder from latent world model state z to block pose."""
    def __init__(self, latent_dim=128, pose_dim=3, hidden=256, depth=2):
        super().__init__()
        layers = []
        d = latent_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, pose_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def evaluate_decoder_transfer(decoder, dataloader, device="cuda"):
    decoder.eval()
    total_mse = 0.0
    count = 0
    criterion = nn.MSELoss(reduction='sum')
    
    with torch.no_grad():
        for z, target_pose in dataloader:
            z = z.to(device)
            target_pose = target_pose.to(device)
            pred_pose = decoder(z)
            total_mse += criterion(pred_pose, target_pose).item()
            count += z.size(0)
            
    return total_mse / max(count, 1)
