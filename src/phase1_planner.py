# ============================================================
# Phase 1 — Planner repair and exact dynamics evaluation (Step 0)
# ============================================================
import os
import json
import time
import numpy as np
import pymunk

try:
    import gymnasium as gym
except ImportError:
    import gym
import gym_pusht

PHASE1_SEEDS = list(range(7000, 7050))
PHASE1_T_MAX = 300
PHASE1_REPLAN = 4
PHASE1_H = 8
PHASE1_POP = 256
PHASE1_ITERS = 4
PHASE1_ELITE = 32
PHASE1_LAMBDA_POSE = 1.0
PHASE1_SIGMA_INIT = 0.3
PHASE1_ACTION_LO = 0.0
PHASE1_ACTION_HI = 512.0
PHASE1_COVERAGE_THRESHOLD = 0.95


def make_env(seed=None):
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", render_mode="rgb_array")
    if seed is not None:
        env.reset(seed=seed)
    return env


def extract_agent_pos(obs, info, env):
    if isinstance(info, dict):
        for k in ("agent_pos", "agent_position", "agent"):
            if k in info:
                return np.asarray(info[k], dtype=np.float32)
    u = env.unwrapped
    if hasattr(u, "agent") and hasattr(u.agent, "position"):
        return np.asarray(u.agent.position, dtype=np.float32)
    if hasattr(u, "agent_pos"):
        return np.asarray(u.agent_pos, dtype=np.float32)
    raise RuntimeError("agent position unavailable")


def extract_block_pose(info, env):
    if isinstance(info, dict):
        for k in ("block_pose", "block_position", "block"):
            if k in info:
                return np.asarray(info[k], dtype=np.float32)
    u = env.unwrapped
    if hasattr(u, "block"):
        b = u.block
        return np.array([b.position.x, b.position.y, float(b.angle)], dtype=np.float32)
    raise RuntimeError("block pose unavailable")


def _snapshot_pymunk_space(env):
    sp = env.unwrapped.space
    saved_bodies = []
    for b in sp.bodies:
        saved_bodies.append({
            "body": b,
            "position": (b.position.x, b.position.y),
            "velocity": (b.velocity.x, b.velocity.y),
            "angle": b.angle,
            "angular_velocity": b.angular_velocity,
            "force": (b.force.x, b.force.y),
            "torque": b.torque,
        })
    return {"bodies": saved_bodies, "space": sp}


def _restore_pymunk_space(snap):
    for s in snap["bodies"]:
        b = s["body"]
        b.position = pymunk.Vec2d(*s["position"])
        b.velocity = pymunk.Vec2d(*s["velocity"])
        b.angle = s["angle"]
        b.angular_velocity = s["angular_velocity"]
        b.force = pymunk.Vec2d(*s["force"])
        b.torque = s["torque"]

    sp = snap["space"]
    for c in sp.constraints:
        if hasattr(c, "impulse"):
            try:
                c.impulse = pymunk.Vec2d(0, 0)
            except Exception:
                pass
    if hasattr(sp, "_cached_arbiters"):
        try:
            sp._cached_arbiters.clear()
        except Exception:
            pass
    if hasattr(sp, "_arbiters"):
        try:
            sp._arbiters.clear()
        except Exception:
            pass


def rollout_exact(env, actions, H):
    snap = _snapshot_pymunk_space(env)
    last_reward = 0.0
    last_block = None
    last_agent = None
    try:
        for t in range(H):
            a = np.clip(actions[t], PHASE1_ACTION_LO, PHASE1_ACTION_HI).astype(np.float32)
            _, reward, terminated, truncated, _ = env.step(a)
            last_reward = float(reward)
            last_block = extract_block_pose({}, env)
            last_agent = extract_agent_pos(None, {}, env)
            if terminated or truncated:
                break
    finally:
        _restore_pymunk_space(snap)
    return {"terminal_reward": last_reward, "terminal_block": last_block, "terminal_agent": last_agent}


def make_cost(cost_kind, goal_pose=(256.0, 256.0, np.pi / 4), lam=PHASE1_LAMBDA_POSE):
    goal = np.array(goal_pose, dtype=np.float32)

    def cost(r):
        cov = r["terminal_reward"]
        if cost_kind in ("coverage", "hybrid"):
            return -cov
        if cost_kind == "terminal_pose":
            b = r["terminal_block"]
            dpos = float(np.linalg.norm(b[:2] - goal[:2]))
            dth = float(abs((b[2] - goal[2] + np.pi) % (2 * np.pi) - np.pi))
            return dpos ** 2 + (dth * 50.0) ** 2
        raise ValueError(cost_kind)
    return cost


def cem_plan(env, cost_fn, rng, H=PHASE1_H, pop=PHASE1_POP, iters=PHASE1_ITERS, elite=PHASE1_ELITE, sigma_init=PHASE1_SIGMA_INIT, warm_start=None):
    arange = PHASE1_ACTION_HI - PHASE1_ACTION_LO
    if warm_start is None:
        mean = np.full((H, 2), PHASE1_ACTION_HI / 2, dtype=np.float32)
    else:
        mean = np.tile(np.asarray(warm_start, dtype=np.float32), (H, 1))

    sigma = np.full((H, 2), sigma_init * arange, dtype=np.float32)
    best_actions = mean.copy()
    best_cost = np.inf

    for _ in range(iters):
        cands = rng.normal(mean, sigma, size=(pop, H, 2)).astype(np.float32)
        cands = np.clip(cands, PHASE1_ACTION_LO, PHASE1_ACTION_HI)
        costs = np.empty(pop)
        for i in range(pop):
            r = rollout_exact(env, cands[i], H)
            costs[i] = cost_fn(r)
        idx = np.argsort(costs)[:elite]
        elite_set = cands[idx]
        mean = elite_set.mean(axis=0)
        sigma = elite_set.std(axis=0) + 1e-3
        if costs[idx[0]] < best_cost:
            best_cost = costs[idx[0]]
            best_actions = cands[idx[0]].copy()
    return best_actions, best_cost
