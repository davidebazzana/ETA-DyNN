"""Search for the most *balanced* values of the weighting factors
gamma_s_1, gamma_s_2, gamma_d_1, gamma_d_2.

Background
----------
In the paper the local-stage cost and the delegation cost are each a weighted
sum of two components (Eq. 1 and Eq. 2):

    c_s = gamma_s_1 * exp(-z')              + gamma_s_2 * (2/pi)*arctan(m/b)
            \_______ term S1 ______/            \_________ term S2 _________/

    c_d = gamma_d_1 * (1 - m)               + gamma_d_2 * exp(t_s - t_S)
            \____ term D1 ____/                 \______ term D2 _______/

The paper fixes every gamma to 0.5 "to unbias with respect to the cost
components".  Setting equal *coefficients* only unbiases the cost if the two
components share the same scale.  They do not: e.g. exp(-z') lives around
~0.5-0.8 while (2/pi)arctan(m/b) lives around ~0.0-0.4, so with 0.5/0.5 the
first term silently dominates c_s.

"Most balanced" is therefore taken to mean: the gamma values for which the two
components of each cost deliver, on average over the operating state space, the
*same contribution* to that cost.  With the normalisation gamma_1 + gamma_2 = 1
(so the overall cost scale matches the 0.5/0.5 baseline) the balance condition

    gamma_1 * E[T1] = gamma_2 * E[T2]

has the closed-form solution

    gamma_1 = E[T2] / (E[T1] + E[T2]),   gamma_2 = E[T1] / (E[T1] + E[T2]).

We compute this two ways -- by the mean magnitude of each term and by the
spread (std) each term injects into the cost -- and we confirm both with an
explicit grid search that minimises the contribution imbalance.  We also report
a decision-level cross-check (fraction of local vs delegate over the grid).

NOTE: this script only *reads* the model; it changes no project file.  The
agent's gamma attributes are set in place purely to evaluate the formulas.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment_aware_task_allocation.agent import Agent
from environment_aware_task_allocation.battery import Battery
from environment_aware_task_allocation.memory import Memory

OUT_DIR = Path(__file__).parent

# --------------------------------------------------------------------------
# Fixed hardware / scenario context
# --------------------------------------------------------------------------
battery = Battery(battery_capacity=200, initial_soc=0.5)
memory = Memory(max_tasks=2000)

energy_costs = {
    "titan": {"stage": 0.3539396298521146, "idle": 0.2687317150290075, "delegation": 0.056629709442887354},
    "jetson": {"stage": 0.06556391602883709, "idle": 0.016505677295133903, "delegation": 0.056629709442887354},
}
device = "jetson"
ghis_prediction = 250 * np.ones(360)
N_STAGES = 2


def make_agent(gammas):
    return Agent(
        battery=battery, memory=memory, dt=10, n_stages=N_STAGES, future_time_window=360,
        stage_energy_cost=energy_costs[device]["stage"],
        delegation_energy_cost=energy_costs[device]["delegation"],
        idle_energy_cost=energy_costs[device]["idle"],
        gamma_s_1=gammas["gamma_s_1"], gamma_s_2=gammas["gamma_s_2"],
        gamma_d_1=gammas["gamma_d_1"], gamma_d_2=gammas["gamma_d_2"],
        test=True,
    )


# --------------------------------------------------------------------------
# 1. Build a representative state grid.
#
# The decision is weight-driven only when the SOC is inside the optimal range
# (outside it the barrier functions force the action regardless of the gammas).
# We therefore measure the balance over b in [0.4, 0.8], full memory range and
# both local stages t_s in {0, 1}.
# --------------------------------------------------------------------------
soc_values = np.linspace(0.4, 0.8, 21)
mem_values = np.linspace(0.0, 1.0, 21)
t_s_values = [0, 1]

states = [(b, m, t_s)
          for b in soc_values
          for m in mem_values
          for t_s in t_s_values]
print(f"Evaluating {len(states)} states "
      f"(SOC in [0.4,0.8], memory in [0,1], t_s in {{0,1}})")


def extract_terms():
    """Return a DataFrame with, for every state, the *unweighted* value of each
    of the four cost components.  We recover each term by evaluating the agent
    cost function with the corresponding gamma set to 1 and the other to 0 --
    this reuses the exact formulas in agent.py instead of duplicating them."""
    agent = make_agent({"gamma_s_1": 1.0, "gamma_s_2": 1.0,
                         "gamma_d_1": 1.0, "gamma_d_2": 1.0})
    rows = []
    for b, m, t_s in states:
        agent.t_s = t_s
        agent.battery.set_soc(b)
        agent.memory.set_usage_perc(m)

        # term S1 = exp(-z')      -> gamma_s_1=1, gamma_s_2=0
        agent.gamma_s_1, agent.gamma_s_2 = 1.0, 0.0
        T_s1 = agent.stage_cost(test=True)
        # term S2 = (2/pi)arctan(m/b) -> gamma_s_1=0, gamma_s_2=1
        agent.gamma_s_1, agent.gamma_s_2 = 0.0, 1.0
        T_s2 = agent.stage_cost(test=True)

        # term D1 = (1 - m)       -> gamma_d_1=1, gamma_d_2=0
        agent.gamma_d_1, agent.gamma_d_2 = 1.0, 0.0
        T_d1 = agent.delegation_cost()
        # term D2 = exp(t_s - t_S) -> gamma_d_1=0, gamma_d_2=1
        agent.gamma_d_1, agent.gamma_d_2 = 0.0, 1.0
        T_d2 = agent.delegation_cost()

        rows.append({"battery_soc": b, "memory_usage": m, "t_s": t_s,
                     "T_s1": T_s1, "T_s2": T_s2, "T_d1": T_d1, "T_d2": T_d2})
    return pd.DataFrame(rows)


terms = extract_terms()
terms.to_csv(OUT_DIR / "gamma_balance_terms.csv", index=False)

stats = terms[["T_s1", "T_s2", "T_d1", "T_d2"]].agg(["mean", "std", "min", "max"]).T
print("\nUnweighted cost-component statistics over the state grid:")
print(stats)


# --------------------------------------------------------------------------
# 2. Closed-form balanced gammas (normalised so gamma_1 + gamma_2 = 1).
#    - mean criterion:  gamma_1 * E[T1] == gamma_2 * E[T2]
#    - std criterion :  gamma_1 * std[T1] == gamma_2 * std[T2]
# --------------------------------------------------------------------------
def balanced_pair(stat1, stat2):
    """gamma weighting the two terms so their contributions are equal."""
    total = stat1 + stat2
    if total == 0:
        return 0.5, 0.5
    g1 = stat2 / total
    g2 = stat1 / total
    return g1, g2


def rms(col):
    """Root-mean-square magnitude == sqrt(E[T^2]) == sqrt(mean^2 + std^2).
    A single scalar that folds in *both* the average level (mean) and the
    variability (std) of a component."""
    return float(np.sqrt((terms[col] ** 2).mean()))


mean_s1, mean_s2 = terms["T_s1"].mean(), terms["T_s2"].mean()
mean_d1, mean_d2 = terms["T_d1"].mean(), terms["T_d2"].mean()
std_s1, std_s2 = terms["T_s1"].std(), terms["T_s2"].std()
std_d1, std_d2 = terms["T_d1"].std(), terms["T_d2"].std()
rms_s1, rms_s2 = rms("T_s1"), rms("T_s2")
rms_d1, rms_d2 = rms("T_d1"), rms("T_d2")

gs1_mean, gs2_mean = balanced_pair(mean_s1, mean_s2)
gd1_mean, gd2_mean = balanced_pair(mean_d1, mean_d2)
gs1_std, gs2_std = balanced_pair(std_s1, std_s2)
gd1_std, gd2_std = balanced_pair(std_d1, std_d2)
# RMS == "both" criterion: balances E[T^2], i.e. mean and std jointly.
gs1_rms, gs2_rms = balanced_pair(rms_s1, rms_s2)
gd1_rms, gd2_rms = balanced_pair(rms_d1, rms_d2)

print("\nBalanced gammas (normalised to sum 1):")
print(f"  [mean criterion] gamma_s_1={gs1_mean:.4f}, gamma_s_2={gs2_mean:.4f} | "
      f"gamma_d_1={gd1_mean:.4f}, gamma_d_2={gd2_mean:.4f}")
print(f"  [std  criterion] gamma_s_1={gs1_std:.4f}, gamma_s_2={gs2_std:.4f} | "
      f"gamma_d_1={gd1_std:.4f}, gamma_d_2={gd2_std:.4f}")
print(f"  [rms  criterion] gamma_s_1={gs1_rms:.4f}, gamma_s_2={gs2_rms:.4f} | "
      f"gamma_d_1={gd1_rms:.4f}, gamma_d_2={gd2_rms:.4f}    (mean+std combined)")


# --------------------------------------------------------------------------
# 3. Explicit grid search confirmation.
#    For each cost we sweep gamma_1 in (0,1), set gamma_2 = 1 - gamma_1 and
#    measure the imbalance of the *mean* contribution of the two terms:
#        imbalance = | mean(gamma_1 T1) - mean(gamma_2 T2) |
#                    -------------------------------------
#                       mean(gamma_1 T1) + mean(gamma_2 T2)
#    in [0, 1]; 0 == perfectly balanced.
# --------------------------------------------------------------------------
grid = np.linspace(0.001, 0.999, 999)


def imbalance_stat(s1, s2):
    """Relative imbalance of the two weighted contributions as gamma_1 sweeps,
    using a scalar magnitude (mean, std or rms) for each component."""
    c1 = grid * s1
    c2 = (1 - grid) * s2
    return np.abs(c1 - c2) / (c1 + c2)


# Imbalance curves for each single-moment criterion ...
imb_s_mean = imbalance_stat(mean_s1, mean_s2)
imb_d_mean = imbalance_stat(mean_d1, mean_d2)
imb_s_std = imbalance_stat(std_s1, std_s2)
imb_d_std = imbalance_stat(std_d1, std_d2)
imb_s_rms = imbalance_stat(rms_s1, rms_s2)
imb_d_rms = imbalance_stat(rms_d1, rms_d2)

gs1_grid = grid[int(np.argmin(imb_s_mean))]       # confirms the mean closed form
gd1_grid = grid[int(np.argmin(imb_d_mean))]
gs1_grid_rms = grid[int(np.argmin(imb_s_rms))]    # confirms the rms closed form
gd1_grid_rms = grid[int(np.argmin(imb_d_rms))]

print("\nGrid-search optimum (mean contribution imbalance):")
print(f"  gamma_s_1={gs1_grid:.4f} (gamma_s_2={1-gs1_grid:.4f}), "
      f"min imbalance={imb_s_mean.min():.2e}")
print(f"  gamma_d_1={gd1_grid:.4f} (gamma_d_2={1-gd1_grid:.4f}), "
      f"min imbalance={imb_d_mean.min():.2e}")
print("\nGrid-search optimum (rms = combined mean+std imbalance):")
print(f"  gamma_s_1={gs1_grid_rms:.4f} (gamma_s_2={1-gs1_grid_rms:.4f})   "
      f"[closed form {gs1_rms:.4f}]")
print(f"  gamma_d_1={gd1_grid_rms:.4f} (gamma_d_2={1-gd1_grid_rms:.4f})   "
      f"[closed form {gd1_rms:.4f}]")


# --------------------------------------------------------------------------
# 4. Decision-level cross-check.
#    Recompute the local/delegate split over the grid for (a) the paper's
#    0.5/0.5 baseline and (b) the recommended balanced gammas.  A balanced
#    parameterisation should also leave the controller able to choose either
#    action rather than collapsing to one.
# --------------------------------------------------------------------------
def decision_split(gammas):
    agent = make_agent(gammas)
    local = delegate = 0
    for b, m, t_s in states:
        agent.t_s = t_s
        agent.battery.set_soc(b)
        agent.memory.set_usage_perc(m)
        battery_discount = agent.battery_discount(ghis_prediction)
        memory_discount = agent.memory_discount(ghis_prediction)
        discount = battery_discount * (1 - memory_discount)
        cs = agent.stage_cost(test=True)
        cd = agent.delegation_cost()
        bl = agent.barrier_function("left")
        br = agent.barrier_function("right")
        if not (np.isfinite(bl) and np.isfinite(br)):
            # exact SOC barrier boundary -> action is forced, not weight-driven
            continue
        local_term = (1 - discount) * cs * bl
        delegate_term = discount * cd * br
        if local_term < delegate_term:
            local += 1
        else:
            delegate += 1
    n = local + delegate
    return local / n, delegate / n


recommended = {
    "gamma_s_1": round(gs1_mean, 3), "gamma_s_2": round(gs2_mean, 3),
    "gamma_d_1": round(gd1_mean, 3), "gamma_d_2": round(gd2_mean, 3),
}
recommended_rms = {
    "gamma_s_1": round(gs1_rms, 3), "gamma_s_2": round(gs2_rms, 3),
    "gamma_d_1": round(gd1_rms, 3), "gamma_d_2": round(gd2_rms, 3),
}
baseline = {"gamma_s_1": 0.5, "gamma_s_2": 0.5, "gamma_d_1": 0.5, "gamma_d_2": 0.5}

base_local, base_deleg = decision_split(baseline)
rec_local, rec_deleg = decision_split(recommended)
rms_local, rms_deleg = decision_split(recommended_rms)
print("\nDecision split over the grid (local fraction / delegate fraction):")
print(f"  baseline 0.5/0.5  : local={base_local:.3f}, delegate={base_deleg:.3f}")
print(f"  recommended (mean): local={rec_local:.3f}, delegate={rec_deleg:.3f}")
print(f"  recommended (rms) : local={rms_local:.3f}, delegate={rms_deleg:.3f}")


# --------------------------------------------------------------------------
# 5. Persist a tidy summary of every candidate.
# --------------------------------------------------------------------------
summary = pd.DataFrame([
    {"criterion": "paper_baseline",
     "gamma_s_1": 0.5, "gamma_s_2": 0.5, "gamma_d_1": 0.5, "gamma_d_2": 0.5},
    {"criterion": "balanced_mean",
     "gamma_s_1": gs1_mean, "gamma_s_2": gs2_mean,
     "gamma_d_1": gd1_mean, "gamma_d_2": gd2_mean},
    {"criterion": "balanced_std",
     "gamma_s_1": gs1_std, "gamma_s_2": gs2_std,
     "gamma_d_1": gd1_std, "gamma_d_2": gd2_std},
    {"criterion": "balanced_rms",
     "gamma_s_1": gs1_rms, "gamma_s_2": gs2_rms,
     "gamma_d_1": gd1_rms, "gamma_d_2": gd2_rms},
    {"criterion": "balanced_grid",
     "gamma_s_1": gs1_grid, "gamma_s_2": 1 - gs1_grid,
     "gamma_d_1": gd1_grid, "gamma_d_2": 1 - gd1_grid},
])
summary.to_csv(OUT_DIR / "gamma_balance_summary.csv", index=False)
print("\nSummary of candidate gamma sets:")
print(summary.to_string(index=False))


# --------------------------------------------------------------------------
# 6. Plots.
# --------------------------------------------------------------------------
# 6a. Imbalance curves for each criterion, with the located optima.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
panels = [
    (axes[0], imb_s_mean, imb_s_std, imb_s_rms,
     gs1_mean, gs1_std, gs1_rms, "stage cost  c_s", ("gamma_s_1", "gamma_s_2")),
    (axes[1], imb_d_mean, imb_d_std, imb_d_rms,
     gd1_mean, gd1_std, gd1_rms, "delegation cost  c_d", ("gamma_d_1", "gamma_d_2")),
]
for ax, im_m, im_s, im_r, g_m, g_s, g_r, name, labels in panels:
    ax.plot(grid, im_m, color="#4C72B0", label="mean criterion")
    ax.plot(grid, im_s, color="#DD8452", label="std criterion")
    ax.plot(grid, im_r, color="#55A868", label="rms (mean+std)")
    ax.axvline(g_r, color="#55A868", linestyle="--",
               label=f"rms {labels[0]}={g_r:.3f}")
    ax.axvline(0.5, color="grey", linestyle=":", label="paper 0.5/0.5")
    ax.set_xlabel(f"{labels[0]}   (= 1 - {labels[1]})")
    ax.set_ylabel("relative contribution imbalance")
    ax.set_title(name)
    ax.legend(fontsize=8)
plt.suptitle("Balancing the two components of each cost "
             "(mean, std and combined rms criteria)", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "gamma_balance_imbalance.png", dpi=150)
plt.close(fig)

# 6b. Mean contribution of each term: baseline vs balanced.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (g1b, g2b, t1, t2, title, names) in zip(axes, [
    (gs1_mean, gs2_mean, "T_s1", "T_s2", "stage cost  c_s",
     ("exp(-z')", "(2/pi)arctan(m/b)")),
    (gd1_mean, gd2_mean, "T_d1", "T_d2", "delegation cost  c_d",
     ("1 - m", "exp(t_s - t_S)")),
]):
    base = [0.5 * terms[t1].mean(), 0.5 * terms[t2].mean()]
    bal = [g1b * terms[t1].mean(), g2b * terms[t2].mean()]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, base, w, label="paper 0.5/0.5", color="#8C8C8C")
    ax.bar(x + w / 2, bal, w, label="balanced", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean contribution to cost")
    ax.set_title(title)
    ax.legend()
plt.suptitle("Mean contribution of each cost component", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "gamma_balance_contributions.png", dpi=150)
plt.close(fig)

print("\nDone. Files written to:", OUT_DIR)
for f in ["gamma_balance_terms.csv", "gamma_balance_summary.csv",
          "gamma_balance_imbalance.png", "gamma_balance_contributions.png"]:
    print(" -", f)

print("\n==> RECOMMENDED balanced gammas (sum-to-1):")
print(f"    mean criterion       : gamma_s_1={gs1_mean:.3f}, gamma_s_2={gs2_mean:.3f}, "
      f"gamma_d_1={gd1_mean:.3f}, gamma_d_2={gd2_mean:.3f}")
print(f"    rms criterion (both) : gamma_s_1={gs1_rms:.3f}, gamma_s_2={gs2_rms:.3f}, "
      f"gamma_d_1={gd1_rms:.3f}, gamma_d_2={gd2_rms:.3f}")
