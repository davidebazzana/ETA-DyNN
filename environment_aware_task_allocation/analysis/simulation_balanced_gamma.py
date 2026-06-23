"""Re-run the Section III.C application-scenario simulation of the ETA-DyNN paper
with the *balanced* gamma weights recommended by gamma_balance_report.pdf instead
of the paper's uniform 0.5/0.5 choice.

Section III.C scenario (paper, page 5)
--------------------------------------
Future time window T = 360, x_MAX = 360, m_MAX = 2000, t_S = 2, x = 260,
current stage t_s = 1.  Three states are examined:

  S1: m = 0.9, b = 0.5  -> delegate          (paper: c_s=0.75, c_d=0.23, d=0)
  S2: m = 0.1, b = 0.5  -> local stage        (paper: c_s=0.47, c_d=0.63, d=0.77)
  S3: m = 0.1, b = 0.3  -> delegate via b_l    (paper: c_s=0.51, c_d=0.63, d=0.78)

Paper baseline weights:   gamma_s_1=gamma_s_2=gamma_d_1=gamma_d_2 = 0.5
Recommended (rms) weights (gamma_balance_report.pdf, Eq. 7):
        gamma_s_1 = 0.402, gamma_s_2 = 0.598,
        gamma_d_1 = 0.322, gamma_d_2 = 0.678

The recommended values are read from gamma_balance_summary.csv (criterion
`balanced_rms`) when that file is present, falling back to the literals above.

Notes
-----
* The cost functions c_s (Eq. 2) and c_d (Eq. 1) are evaluated through the
  Agent object in *test* mode, exactly as gamma_balance_search.py does, so the
  script reuses the simulator's own formulas rather than re-implementing them.
* The discount d (Eqs. 5-7) and the barrier functions b_l, b_u (Eqs. 8-9) do
  NOT depend on the gamma weights, so they are identical to the paper.  The
  discount is computed here in closed form from the scenario parameters
  (x, x_MAX, m_MAX and the memory fraction m, with m_t = m * m_MAX).
* This script only *reads* the model; it modifies no project file.

Run with:
    python -m environment_aware_task_allocation.simulation_balanced_gamma
"""

from pathlib import Path

import numpy as np
import pandas as pd

from environment_aware_task_allocation.agent import Agent
from environment_aware_task_allocation.battery import Battery
from environment_aware_task_allocation.memory import Memory

OUT_DIR = Path(__file__).parent

# --------------------------------------------------------------------------
# Section III.C global scenario parameters.
# --------------------------------------------------------------------------
T = 360            # future time window
X_MAX = 360        # max expected tasks if irradiance held at 1000 W/m^2
M_MAX = 2000       # memory capacity (number of tasks)
T_S = 2            # maximum number of EE stages (t_S)  -> n_stages
X = 260            # expected number of tasks x(t)
T_S_NOW = 1        # current stage t_s

# Per-scenario state: (label, memory usage m, battery SOC b, L, expected decision)
SCENARIOS = [
    ("S1", 0.9, 0.5, 3072, "delegate"),
    ("S2", 0.1, 0.5, 3072, "local stage"),
    ("S3", 0.1, 0.3, 1024, "delegate (barrier b_l)"),
]

# --------------------------------------------------------------------------
# Gamma sets.
# --------------------------------------------------------------------------
PAPER_BASELINE = {
    "gamma_s_1": 0.5, "gamma_s_2": 0.5, "gamma_d_1": 0.5, "gamma_d_2": 0.5,
}

# Recommended rms-balanced weights (gamma_balance_report.pdf, Eq. 7).
RECOMMENDED_RMS_FALLBACK = {
    "gamma_s_1": 0.402, "gamma_s_2": 0.598, "gamma_d_1": 0.322, "gamma_d_2": 0.678,
}


def load_recommended_gammas():
    """Read the balanced (rms) weights from gamma_balance_summary.csv if available;
    otherwise fall back to the literals reported in the gamma-balance report."""
    summary_path = OUT_DIR / "gamma_balance_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        row = df.loc[df["criterion"] == "balanced_rms"]
        if not row.empty:
            r = row.iloc[0]
            return {
                "gamma_s_1": float(r["gamma_s_1"]), "gamma_s_2": float(r["gamma_s_2"]),
                "gamma_d_1": float(r["gamma_d_1"]), "gamma_d_2": float(r["gamma_d_2"]),
            }, str(summary_path.name)
    return dict(RECOMMENDED_RMS_FALLBACK), "literals (Eq. 7 of the report)"


# --------------------------------------------------------------------------
# Hardware / scenario context (same values used in gamma_balance_search.py).
# Only the cost formulas are exercised, so these values do not influence the
# reported c_s / c_d magnitudes beyond setting m_MAX = 2000 and t_S = 2.
# --------------------------------------------------------------------------
ENERGY_COSTS = {"stage": 0.06556391602883709,
                "idle": 0.016505677295133903,
                "delegation": 0.056629709442887354}


def make_agent(gammas):
    battery = Battery(battery_capacity=200, initial_soc=0.5)
    memory = Memory(max_tasks=M_MAX)
    return Agent(
        battery=battery, memory=memory, dt=10, n_stages=T_S, future_time_window=T,
        stage_energy_cost=ENERGY_COSTS["stage"],
        delegation_energy_cost=ENERGY_COSTS["delegation"],
        idle_energy_cost=ENERGY_COSTS["idle"],
        gamma_s_1=gammas["gamma_s_1"], gamma_s_2=gammas["gamma_s_2"],
        gamma_d_1=gammas["gamma_d_1"], gamma_d_2=gammas["gamma_d_2"],
        test=True,
    )


def discount(m, L):
    """Overall discount d = d_b * (1 - d_m)  (paper Eqs. 5-7).

    d_b  (battery discount, Eq. 5):
        1                          if x <= L
        (x_MAX - x)/(x_MAX - L)    if L < x <= x_MAX
        0                          if x > x_MAX
    d_m  (memory discount, Eq. 6):
        min((m_t + x)/m_MAX, 1),   with m_t = m * m_MAX (the tasks in memory).
    """
    if X <= L:
        d_b = 1.0
    elif X <= X_MAX:
        d_b = (X_MAX - X) / (X_MAX - L)
    else:
        d_b = 0.0

    m_t = m * M_MAX
    d_m = min((m_t + X) / M_MAX, 1.0)
    return d_b * (1.0 - d_m), d_b, d_m


def evaluate(gammas):
    """Evaluate c_s, c_d, discount, barriers and the resulting decision for every
    scenario under the given gamma set."""
    agent = make_agent(gammas)
    rows = []
    for label, m, b, L, expected in SCENARIOS:
        agent.t_s = T_S_NOW
        agent.battery.set_soc(b)
        agent.memory.set_usage_perc(m)

        c_s = agent.stage_cost(test=True)
        c_d = agent.delegation_cost()
        # Outside the optimal SOC range the barrier argument is negative, so the
        # log returns nan by design (it signals a barrier-forced action below).
        with np.errstate(invalid="ignore"):
            b_l = agent.barrier_function("left")    # -ln(b - 0.4)
            b_u = agent.barrier_function("right")   # -ln(0.8 - b)
        d, d_b, d_m = discount(m, L)

        # Minimisation (paper Eq. 10) with barrier-forced actions.
        if np.isnan(b_l):
            # SOC below the optimal lower barrier -> forced delegation.
            decision = "delegate"
            local_term = np.nan
            delegate_term = np.nan
        elif np.isnan(b_u):
            # SOC above the optimal upper barrier -> forced local stage.
            decision = "local stage"
            local_term = np.nan
            delegate_term = np.nan
        else:
            local_term = (1 - d) * c_s * b_l
            delegate_term = d * c_d * b_u
            decision = "local stage" if local_term < delegate_term else "delegate"

        rows.append({
            "scenario": label, "m": m, "b": b, "L": L,
            "c_s": c_s, "c_d": c_d, "d_b": d_b, "d_m": d_m, "d": d,
            "b_l": b_l, "b_u": b_u,
            "local_term": local_term, "delegate_term": delegate_term,
            "decision": decision, "paper_decision": expected,
        })
    return pd.DataFrame(rows)


def fmt(x):
    return "   nan" if (isinstance(x, float) and np.isnan(x)) else f"{x:6.3f}"


def print_table(title, gammas, df):
    print(f"\n{'='*78}\n{title}")
    print(f"  gammas: s1={gammas['gamma_s_1']:.3f}  s2={gammas['gamma_s_2']:.3f}  "
          f"d1={gammas['gamma_d_1']:.3f}  d2={gammas['gamma_d_2']:.3f}")
    print(f"{'-'*78}")
    print(f"  {'sc':>3}  {'m':>4}  {'b':>4}  {'c_s':>6}  {'c_d':>6}  {'d':>6}  "
          f"{'b_l':>6}  {'b_u':>6}  decision")
    for _, r in df.iterrows():
        print(f"  {r['scenario']:>3}  {r['m']:>4}  {r['b']:>4}  "
              f"{fmt(r['c_s'])}  {fmt(r['c_d'])}  {fmt(r['d'])}  "
              f"{fmt(r['b_l'])}  {fmt(r['b_u'])}  {r['decision']}  "
              f"(paper: {r['paper_decision']})")


def main():
    recommended, src = load_recommended_gammas()

    print("ETA-DyNN Section III.C simulation")
    print(f"  T={T}, x_MAX={X_MAX}, m_MAX={M_MAX}, t_S={T_S}, x={X}, t_s={T_S_NOW}")
    print(f"  recommended (balanced-rms) gammas read from: {src}")

    df_base = evaluate(PAPER_BASELINE)
    df_rec = evaluate(recommended)

    print_table("PAPER BASELINE (gamma = 0.5 / 0.5 / 0.5 / 0.5)", PAPER_BASELINE, df_base)
    print_table("RECOMMENDED BALANCED (rms) GAMMAS", recommended, df_rec)

    # Side-by-side comparison of the cost magnitudes (decisions / discount are
    # unchanged because the discount and barriers do not depend on the gammas).
    print(f"\n{'='*78}\nCOMPARISON (baseline -> balanced)")
    print(f"{'-'*78}")
    print(f"  {'sc':>3}  {'c_s base':>9}  {'c_s bal':>9}  {'c_d base':>9}  "
          f"{'c_d bal':>9}  {'d':>6}  decision (base -> bal)")
    for (_, rb), (_, rr) in zip(df_base.iterrows(), df_rec.iterrows()):
        print(f"  {rb['scenario']:>3}  {rb['c_s']:>9.3f}  {rr['c_s']:>9.3f}  "
              f"{rb['c_d']:>9.3f}  {rr['c_d']:>9.3f}  {rb['d']:>6.3f}  "
              f"{rb['decision']} -> {rr['decision']}")

    # Persist a tidy CSV.
    df_base["gamma_set"] = "paper_baseline"
    df_rec["gamma_set"] = "balanced_rms"
    out = pd.concat([df_base, df_rec], ignore_index=True)
    out_path = OUT_DIR / "simulation_balanced_gamma.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
