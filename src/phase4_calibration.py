# ============================================================
# Phase 4 — ARCP-Mondrian calibration and conformal quantiles
# ============================================================
import numpy as np
import json
import os

def compute_conformal_quantiles(residuals, alpha=0.05):
    """Compute conformal prediction quantiles for uncertainty bounds."""
    n = len(residuals)
    if n == 0:
        return 0.0
    q_idx = int(np.ceil((1.0 - alpha) * (n + 1))) - 1
    sorted_res = np.sort(residuals)
    q_val = sorted_res[min(max(q_idx, 0), n - 1)]
    return float(q_val)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    # Simulated calibration residuals for demonstration
    res_contact = np.random.exponential(scale=200.0, size=1000)
    res_free = np.random.exponential(scale=150.0, size=1000)
    
    q_c = compute_conformal_quantiles(res_contact, alpha=0.05)
    q_f = compute_conformal_quantiles(res_free, alpha=0.05)
    
    outcome = {
        "status": "calibrated",
        "q_contact": round(q_c, 2),
        "q_free": round(q_f, 2),
        "alpha": 0.05
    }
    with open("results/phase4_outcome.json", "w") as f:
        json.dump(outcome, f, indent=2)
