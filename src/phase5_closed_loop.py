# ============================================================
# Phase 5 — Closed-loop runner, Gate analysis & Exploitability Diagnostic
# ============================================================
import numpy as np
import torch

def cem_plan_learned(z0, rng, G_phi, h_task, pose_std, goal_norm, device="cuda", H=8, pop=256, iters=4, elite=32, sigma_init=0.3, action_lo=0.0, action_hi=512.0):
    mean = torch.full((H, 2), 256.0, device=device)
    sigma = torch.full((H, 2), sigma_init * (action_hi - action_lo), device=device)
    best_actions = mean.clone()
    best_cost = float("inf")

    goal_t = torch.from_numpy(goal_norm).float().to(device)

    for _ in range(iters):
        noise = torch.randn(pop, H, 2, device=device, generator=rng)
        cands = (mean[None] + sigma[None] * noise).clamp(action_lo, action_hi)

        z = z0[None].expand(pop, -1).contiguous()
        for t in range(H):
            z = G_phi(z, cands[:, t])
        y_term = h_task(z)

        d_pos = torch.norm(y_term[:, :2] - goal_t[:2], dim=1) * pose_std[0]
        d_theta = torch.abs(y_term[:, 2] - goal_t[2]) * pose_std[2]
        cost = d_pos + 5.0 * d_theta

        idx = cost.argsort()[:elite]
        elite_set = cands[idx]
        mean = elite_set.mean(dim=0)
        sigma = elite_set.std(dim=0) + 1e-3

        if cost[idx[0]].item() < best_cost:
            best_cost = cost[idx[0]].item()
            best_actions = cands[idx[0]].clone()

    return best_actions, best_cost
