# ============================================================
# Phase 0 — Pilot corpus calibration and environment dry-runs
# ============================================================
import os
import json
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym
import gym_pusht

def run_pilot(num_episodes=5, seed=42):
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", render_mode="rgb_array")
    stats = {"episodes": num_episodes, "steps": [], "rewards": []}
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        steps = 0
        while not done and steps < 300:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
        stats["steps"].append(steps)
        stats["rewards"].append(total_reward)
        
    print(f"Pilot completed: average reward = {np.mean(stats['rewards']):.2f}")
    return stats

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    res = run_pilot()
    with open("results/step0_outcome.json", "w") as f:
        json.dump(res, f, indent=2)
