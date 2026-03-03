"""
RL inventory agent: PPO vs classical OR comparison.

This module provides:

  1. ``train_ppo`` — Train a PPO agent on ``InventoryEnv`` using stable-baselines3.
  2. ``evaluate_policy`` — Roll out any policy on a fixed demand sequence.
  3. ``run_comparison`` — Compare PPO, Wagner-Whitin, Silver-Meal, and Lot-for-Lot
     on the same demand episodes and produce a summary DataFrame + plots.
  4. ``ClassicalPolicy`` — Wrapper that lets classical models act as policies inside
     the Gymnasium rollout loop (deterministic, demand-sequence-based).

Framing
-------
    "Classical Operations Research vs. Learning-Based Inventory Control"

    WW provides the lower bound (optimal deterministic cost).
    PPO is trained on stochastic demand episodes.
    The *optimality gap* of each approach is reported as:

        gap(%) = (cost - WW_cost) / WW_cost * 100

Usage
-----
    # Quick comparison (no GPU needed)
    from inventory_optimization.rl.agent import run_comparison
    df, fig = run_comparison(n_episodes=50, timesteps=30_000)
    fig.savefig("rl_comparison.png")
    print(df)
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.figure import Figure
except ImportError as exc:  # pragma: no cover
    raise ImportError("matplotlib and pandas are required: pip install matplotlib pandas") from exc

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover
    raise ImportError("gymnasium is required: pip install gymnasium") from exc

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False
    warnings.warn(
        "stable-baselines3 not found. PPO training disabled. "
        "Classical comparisons still work without it.",
        ImportWarning,
        stacklevel=2,
    )

from inventory_optimization.models import (
    LotForLotModel,
    SilverMealHeuristic,
    WagnerWhitinModel,
)
from inventory_optimization.rl.env import InventoryEnv


# ---------------------------------------------------------------------------
# Classical policy wrapper
# ---------------------------------------------------------------------------


class ClassicalPolicy:
    """Wraps a deterministic classical model as a step-by-step policy.

    At each environment step the policy pre-solves the remaining horizon
    using the *known* demand sequence (requires ``demand_sequence`` in env).

    Parameters
    ----------
    model_cls : type
        A subclass of ``InventoryModel``.
    ordering_cost, unit_cost, carrying_charge : float
        Cost parameters (must match the environment's parameters).
    """

    def __init__(
        self,
        model_cls: type,
        ordering_cost: float,
        unit_cost: float,
        carrying_charge: float,
    ) -> None:
        self.model_cls = model_cls
        self.ordering_cost = ordering_cost
        self.unit_cost = unit_cost
        self.carrying_charge = carrying_charge
        self._plan: Dict[int, float] = {}

    def reset(self, demand_sequence: List[float]) -> None:
        """Pre-solve the full horizon."""
        model = self.model_cls(
            demand_sequence,
            self.ordering_cost,
            self.unit_cost,
            self.carrying_charge,
        )
        _, details = model.calculate_cost()
        self._plan = {
            d["Period"] - 1: d["Replenishment"]
            for d in details
            if d["Replenishment"] > 0
        }

    def predict(self, period: int) -> float:
        """Return the planned order quantity for the given period."""
        return self._plan.get(period, 0.0)


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------


def evaluate_classical(
    model_cls: type,
    demand_sequence: List[float],
    ordering_cost: float,
    unit_cost: float,
    carrying_charge: float,
    backorder_cost: float = 0.0,
) -> Tuple[float, List[float]]:
    """Evaluate a classical model on a fixed demand sequence.

    Returns
    -------
    total_cost : float
    cost_history : list[float]
        Per-period cost (ordering + holding; no backorders for classical models).
    """
    model = model_cls(demand_sequence, ordering_cost, unit_cost, carrying_charge)
    total_cost, details = model.calculate_cost()
    h = carrying_charge * unit_cost
    cost_history = []
    for d in details:
        setup = ordering_cost if d["Replenishment"] > 0 else 0.0
        holding = max(d["Ending_Inventory"], 0.0) * h
        cost_history.append(setup + holding)
    return total_cost, cost_history


def evaluate_ppo(
    model: "PPO",
    env: InventoryEnv,
    demand_sequence: List[float],
    seed: int = 0,
) -> Tuple[float, List[float]]:
    """Roll out a trained PPO model on a fixed demand episode."""
    env_eval = InventoryEnv(
        n_periods=env.n_periods,
        ordering_cost=env.ordering_cost,
        unit_cost=env.unit_cost,
        holding_cost_rate=env._h / env.unit_cost,
        backorder_cost=env.backorder_cost,
        demand_sequence=demand_sequence,
    )
    obs, _ = env_eval.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env_eval.step(action)
        done = terminated or truncated
    return env_eval.get_episode_cost(), env_eval.get_cost_history()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_ppo(
    n_periods: int = 12,
    ordering_cost: float = 160.0,
    unit_cost: float = 5.0,
    carrying_charge: float = 0.1,
    backorder_cost: float = 30.0,
    mean_demand: float = 100.0,
    std_demand: float = 20.0,
    total_timesteps: int = 50_000,
    n_envs: int = 4,
    seed: int = 42,
    verbose: int = 0,
) -> "PPO":
    """Train a PPO agent on stochastic inventory episodes.

    Parameters
    ----------
    total_timesteps : int
        Total environment interaction steps for training.
    n_envs : int
        Number of parallel environments.
    verbose : int
        Verbosity level (0 = silent, 1 = info).

    Returns
    -------
    PPO
        Trained stable-baselines3 PPO model.
    """
    if not _SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 is required for PPO training: pip install stable-baselines3"
        )

    def make_env():
        return InventoryEnv(
            n_periods=n_periods,
            ordering_cost=ordering_cost,
            unit_cost=unit_cost,
            holding_cost_rate=carrying_charge,
            backorder_cost=backorder_cost,
            mean_demand=mean_demand,
            std_demand=std_demand,
        )

    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=seed)
    agent = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=verbose,
        seed=seed,
    )
    agent.learn(total_timesteps=total_timesteps)
    return agent


# ---------------------------------------------------------------------------
# Comparison dashboard
# ---------------------------------------------------------------------------


def run_comparison(
    demand_sequence: Optional[List[float]] = None,
    n_periods: int = 12,
    ordering_cost: float = 160.0,
    unit_cost: float = 5.0,
    carrying_charge: float = 0.1,
    backorder_cost: float = 0.0,
    n_episodes: int = 30,
    timesteps: int = 30_000,
    seed: int = 42,
    train_rl: bool = True,
) -> Tuple["pd.DataFrame", "Figure"]:
    """Run a full comparison of PPO vs. classical models.

    Parameters
    ----------
    demand_sequence : list[float], optional
        Fixed demand for deterministic comparison. If None, uses random sequences.
    n_episodes : int
        Number of random demand episodes to average over (ignored when
        ``demand_sequence`` is provided).
    timesteps : int
        PPO training timesteps (ignored when ``train_rl=False``).
    train_rl : bool
        Whether to train a PPO agent. Set False to compare only classical models.

    Returns
    -------
    results_df : pd.DataFrame
        Per-method summary: mean cost, std cost, mean gap (% vs. WW).
    fig : matplotlib.Figure
        Two-panel figure: (left) cost distributions, (right) mean per-period cost profile.
    """
    rng = np.random.default_rng(seed)
    CLASSICAL = {
        "Wagner-Whitin (Optimal)": WagnerWhitinModel,
        "Silver-Meal": SilverMealHeuristic,
        "Lot-for-Lot": LotForLotModel,
    }

    # Build episode demand sequences
    if demand_sequence is not None:
        episodes = [list(demand_sequence)]
        n_episodes = 1
    else:
        episodes = [
            list(np.clip(rng.normal(100, 20, n_periods), 0, None))
            for _ in range(n_episodes)
        ]

    # Train PPO
    ppo_agent = None
    if train_rl and _SB3_AVAILABLE:
        print(f"Training PPO for {timesteps:,} timesteps...")
        ppo_agent = train_ppo(
            n_periods=n_periods,
            ordering_cost=ordering_cost,
            unit_cost=unit_cost,
            carrying_charge=carrying_charge,
            backorder_cost=backorder_cost,
            mean_demand=100.0,
            std_demand=20.0,
            total_timesteps=timesteps,
            seed=seed,
            verbose=0,
        )
        print("Training complete.")

    # Collect per-episode costs
    all_costs: Dict[str, List[float]] = {name: [] for name in CLASSICAL}
    all_profiles: Dict[str, List[List[float]]] = {name: [] for name in CLASSICAL}
    if ppo_agent is not None:
        all_costs["PPO"] = []
        all_profiles["PPO"] = []

    eval_env = InventoryEnv(
        n_periods=n_periods,
        ordering_cost=ordering_cost,
        unit_cost=unit_cost,
        holding_cost_rate=carrying_charge,
        backorder_cost=backorder_cost,
    )

    for ep_demand in episodes:
        for name, cls in CLASSICAL.items():
            cost, profile = evaluate_classical(
                cls, ep_demand, ordering_cost, unit_cost, carrying_charge
            )
            all_costs[name].append(cost)
            all_profiles[name].append(profile)

        if ppo_agent is not None:
            cost, profile = evaluate_ppo(ppo_agent, eval_env, ep_demand)
            all_costs["PPO"].append(cost)
            all_profiles["PPO"].append(profile)

    # Build results DataFrame
    ww_costs = np.array(all_costs["Wagner-Whitin (Optimal)"])
    rows = []
    for name, costs in all_costs.items():
        arr = np.array(costs)
        gaps = (arr - ww_costs) / np.maximum(ww_costs, 1e-9) * 100
        rows.append(
            {
                "Method": name,
                "Mean Cost": round(float(arr.mean()), 2),
                "Std Cost": round(float(arr.std()), 2),
                "Min Cost": round(float(arr.min()), 2),
                "Max Cost": round(float(arr.max()), 2),
                "Mean Gap vs WW (%)": round(float(gaps.mean()), 2),
            }
        )
    results_df = pd.DataFrame(rows).sort_values("Mean Cost").reset_index(drop=True)

    # Build figure
    fig, (ax_box, ax_profile) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_names = list(all_costs.keys())

    # Box plots of total cost distribution
    data_for_box = [all_costs[m] for m in method_names]
    bp = ax_box.boxplot(data_for_box, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_box.set_xticks(range(1, len(method_names) + 1))
    ax_box.set_xticklabels(
        [m.replace(" ", "\n") for m in method_names], fontsize=8
    )
    ax_box.set_ylabel("Total Cost ($)")
    ax_box.set_title("Cost Distribution by Method")
    ax_box.grid(axis="y", alpha=0.4)

    # Mean per-period cost profile
    for i, name in enumerate(method_names):
        profiles = all_profiles[name]
        if not profiles:
            continue
        max_len = max(len(p) for p in profiles)
        padded = [p + [0.0] * (max_len - len(p)) for p in profiles]
        mean_profile = np.mean(padded, axis=0)
        lw = 2.2 if "Wagner" in name else 1.2
        ls = "-" if "Wagner" in name else "--"
        ax_profile.plot(
            range(1, len(mean_profile) + 1),
            mean_profile,
            label=name,
            color=colors[i % len(colors)],
            linewidth=lw,
            linestyle=ls,
        )

    ax_profile.set_xlabel("Period")
    ax_profile.set_ylabel("Mean Period Cost ($)")
    ax_profile.set_title("Mean Per-Period Cost Profile")
    ax_profile.legend(fontsize=8)
    ax_profile.grid(alpha=0.3)

    fig.suptitle(
        "Classical OR vs. Learning-Based Inventory Control",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    return results_df, fig
