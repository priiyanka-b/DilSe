"""
simulate_wearables.py: Stage 3 of the wearable stream (DECISIONS.md D-13 to D-16).

Gives every validation and test patient a SYNTHETIC 60-day wearable history:
daily resting heart rate, night HRV (RMSSD), sleep hours and steps.
Nothing here is real patient data; every series is generated from recipe.json.

HONESTY RULES (D-13)
  1. The simulation never looks at the patient's ECG label. Only age and sex
     are read (sex sets the resting HR level, D-15).
  2. Half the patients, chosen at random and independently of the label, get
     an injected deterioration episode. The monitor is later scored only on
     finding these injected episodes: a synthetic test, not a clinical result.

ONE MEASURE FOR ONE PATIENT
  value(day) = personal level + weekday effect + wobble(day) + episode(day)
  wobble(day) = phi * wobble(day-1) + noise   (stationary AR(1), recipe.json)
  Steps and HRV are generated on a log scale (multiplicative changes).

DETERIORATION EPISODES (D-16, option A: a grid of sizes = "detection limits")
  severity 1-4 = resting HR rises by +3, +5, +8 or +12 bpm
  ramp         = the rise builds linearly over 3, 5 or 7 days, then holds
  start day    = random day between 36 and 53 (so 35+ clean baseline days)
  The other measures worsen at the same time by fixed, modest amounts per grade
  (SEVERITY_OTHER: HRV -10 to -30%, sleep -0.25 to -1 h, steps -10 to -40%).
  These amounts are SCENARIO DESIGN CHOICES, not clinical facts.

Inputs:  outputs/recipe/recipe.json
         data/processed/meta_val.csv, data/processed/meta_test.csv
Outputs: data/processed/wearables_val.csv, wearables_test.csv   (one row per patient-day)
         data/processed/episodes_val.csv,  episodes_test.csv    (one row per patient)
         outputs/wearables_sim/realism_check.png
         outputs/wearables_sim/example_patients.png
         outputs/wearables_sim/simulation_summary.txt
Run:     python simulate_wearables.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 2026
N_DAYS = 60
EPISODE_SHARE = 0.5
SEVERITY_RHR = {1: 3.0, 2: 5.0, 3: 8.0, 4: 12.0}     # bpm, D-16
# Co-moving changes in the other measures, per severity grade. SCENARIO DESIGN
# CHOICES (not clinical facts), deliberately modest so that resting HR stays the
# main signal and no measure becomes physically implausible.
SEVERITY_OTHER = {          # HRV change, sleep change (h), steps change
    1: (-0.10, -0.25, -0.10),
    2: (-0.15, -0.50, -0.20),
    3: (-0.20, -0.75, -0.30),
    4: (-0.30, -1.00, -0.40),
}
RAMP_DAYS = [3, 5, 7]
START_RANGE = (36, 53)                              # inclusive
SEX_CODE = {0: "male", 1: "female"}                 # inferred coding (D-4 note), still unconfirmed
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

PROC = Path("data/processed")
OUT = Path("outputs/wearables_sim")
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9a94"

log_lines = []


def log(text=""):
    print(text)
    log_lines.append(str(text))


def v(entry):
    return entry["value"]


def ar1(rng, n, phi, sd):
    """Stationary AR(1) series with marginal SD = sd."""
    e = np.empty(n)
    e[0] = rng.normal(0, sd)
    innov = sd * np.sqrt(max(1 - phi ** 2, 1e-6))
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.normal(0, innov)
    return e


def simulate_split(meta, recipe, rng, split):
    L, D = recipe["levels"], recipe["dynamics"]
    rhr_spread = v(D["resting_hr_bpm"]["sd_log_spread"])

    def personal_sd(key):
        spread = D[key]["sd_log_spread"]["value"]
        spread = rhr_spread if spread is None else spread   # HRV stand-in (recipe note)
        return v(D[key]["sd"]) * np.exp(rng.normal(0, spread))

    rows, episodes = [], []
    for p in meta.itertuples():
        sex = SEX_CODE.get(int(p.sex), "male")
        start_dow = int(rng.integers(0, 7))
        dow = (start_dow + np.arange(N_DAYS)) % 7

        # --- personal levels (D-15)
        rhr_level = rng.normal(v(L["resting_hr_bpm"][sex]["mean"]), v(L["resting_hr_bpm"][sex]["sd"]))
        rhr_level = float(np.clip(rhr_level, 40, 110))
        log_steps_level = rng.normal(v(L["steps_per_day"]["mu_log"]), v(L["steps_per_day"]["sigma_log"]))
        sleep_level = float(np.clip(rng.normal(v(L["sleep_hours"]["mean"]), v(L["sleep_hours"]["sd"])), 4, 10))
        log_hrv_level = rng.normal(v(L["hrv_rmssd_ms"]["mu_log"]), v(L["hrv_rmssd_ms"]["sigma_log"]))

        # --- personal wobble sizes and the series without any episode
        sds = {k: personal_sd(k) for k in ["resting_hr_bpm", "hrv_rmssd_ms", "sleep_hours", "steps_per_day"]}
        series = {}
        for key, level in [("resting_hr_bpm", rhr_level), ("hrv_rmssd_ms", log_hrv_level),
                           ("sleep_hours", sleep_level), ("steps_per_day", log_steps_level)]:
            wk = np.array([D[key]["weekday"]["value"][DAYS[d]] for d in dow])
            series[key] = level + wk + ar1(rng, N_DAYS, v(D[key]["phi"]), sds[key])

        # --- deterioration episode (D-16), assigned independently of the label
        has_ep = rng.random() < EPISODE_SHARE
        ep = {"ecg_id": p.ecg_id, "split": split, "episode": int(has_ep),
              "severity": 0, "rhr_rise_bpm": 0.0, "ramp_days": 0, "start_day": -1,
              "age": p.age, "sex": sex}
        if has_ep:
            sev = int(rng.integers(1, 5))
            ramp = int(rng.choice(RAMP_DAYS))
            start = int(rng.integers(START_RANGE[0], START_RANGE[1] + 1))
            shape = np.clip((np.arange(N_DAYS) - start + 1) / ramp, 0, 1)     # 0 before, ramps to 1
            n_sd = SEVERITY_RHR[sev] / sds["resting_hr_bpm"]       # for reporting only
            d_hrv, d_sleep, d_steps = SEVERITY_OTHER[sev]
            series["resting_hr_bpm"] += SEVERITY_RHR[sev] * shape
            series["hrv_rmssd_ms"] += np.log(1 + d_hrv) * shape      # log scale: -20% = log(0.8)
            series["sleep_hours"] += d_sleep * shape
            series["steps_per_day"] += np.log(1 + d_steps) * shape
            ep.update({"severity": sev, "rhr_rise_bpm": SEVERITY_RHR[sev], "ramp_days": ramp,
                       "start_day": start, "size_in_sd": round(float(n_sd), 2)})
        else:
            shape = np.zeros(N_DAYS)
        episodes.append(ep)

        rows.append(pd.DataFrame({
            "ecg_id": p.ecg_id, "day": np.arange(N_DAYS), "weekday": [DAYS[d] for d in dow],
            "resting_hr": np.round(series["resting_hr_bpm"], 1),
            "hrv_rmssd": np.round(np.exp(series["hrv_rmssd_ms"]), 1),
            "sleep_hours": np.round(np.clip(series["sleep_hours"], 0, 14), 2),
            "steps": np.round(np.exp(series["steps_per_day"])).astype(int),
            "in_episode": (shape > 0).astype(int),
        }))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(episodes)


def day_changes(df, col, id_col):
    return df.sort_values([id_col, "day"]).groupby(id_col)[col].diff().dropna()


def lag1(df, col, id_col):
    vals = []
    for _, g in df.groupby(id_col):
        x = g[col].to_numpy()
        if len(x) > 10 and np.std(x) > 0:
            vals.append(np.corrcoef(x[:-1], x[1:])[0, 1])
    return np.median(vals)


def realism_check(sim, pm):
    """Compare clean simulated patients with real PMData on day-to-day behaviour."""
    clean = sim[sim["ecg_id"].isin(sim.groupby("ecg_id")["in_episode"].max().loc[lambda s: s == 0].index)]
    pm = pm.copy()
    pm["day"] = (pm["date"] - pm.groupby("pid")["date"].transform("min")).dt.days
    pm = pm.set_index(["pid", "day"]).groupby(level=0).apply(
        lambda g: g.droplevel(0).reindex(range(g.index.get_level_values(1).max() + 1))).reset_index()
    pm["log_steps"] = np.log(pm["steps"].where(pm["steps"] > 0))
    clean = clean.assign(log_steps=np.log(clean["steps"].clip(lower=1)))

    measures = [("resting_hr", "Resting HR: change from yesterday (bpm)"),
                ("sleep_hours", "Sleep: change from last night (h)"),
                ("log_steps", "Steps: change from yesterday (log)")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    log("\nREALISM CHECK: clean simulated patients vs real PMData (day-to-day behaviour)")
    log(f"{'measure':12s} {'real SD of change':>18s} {'sim SD of change':>17s} {'real memory':>12s} {'sim memory':>11s}")
    for ax, (col, title) in zip(axes, measures):
        r = day_changes(pm, col, "pid")
        s = day_changes(clean, col, "ecg_id")
        lo, hi = np.percentile(np.concatenate([r, s]), [0.5, 99.5])
        if col == "resting_hr":     # simulated values are rounded to 0.1 bpm: align bins to avoid spikes
            bins = np.arange(np.floor(lo) - 0.05, np.ceil(hi) + 0.1, 0.2)
        else:
            bins = np.linspace(lo, hi, 50)
        ax.hist(r, bins=bins, density=True, color=GREY, alpha=0.55, label="Real (PMData)")
        ax.hist(s, bins=bins, density=True, histtype="step", linewidth=2, color=BLUE, label="Simulated")
        ax.set_title(title, loc="left", fontsize=10)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        log(f"{col:12s} {r.std():18.3f} {s.std():17.3f} {lag1(pm.dropna(subset=[col]), col, 'pid'):12.2f} "
            f"{lag1(clean, col, 'ecg_id'):11.2f}")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Day-to-day changes: real vs simulated (simulated patients without an episode)",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "realism_check.png", dpi=140)
    plt.close(fig)
    log("Note: the recipe was fitted on PMData, so this is a consistency check, not independent validation.")


def example_plot(sim, eps):
    stable = eps[eps["episode"] == 0]["ecg_id"].iloc[0]
    sick = eps[(eps["episode"] == 1) & (eps["severity"] == 3)].iloc[0]
    cols = [("resting_hr", "Resting HR (bpm)"), ("hrv_rmssd", "HRV RMSSD (ms)"),
            ("sleep_hours", "Sleep (h)"), ("steps", "Steps")]
    fig, axes = plt.subplots(4, 2, figsize=(13, 9), sharex=True)
    for j, (pid, title) in enumerate([(stable, "No episode"),
                                      (sick["ecg_id"], f"Episode: +{sick['rhr_rise_bpm']:.0f} bpm, "
                                                       f"{sick['ramp_days']}-day ramp from day {sick['start_day']}")]):
        d = sim[sim["ecg_id"] == pid]
        for i, (c, lab) in enumerate(cols):
            ax = axes[i, j]
            ax.plot(d["day"], d[c], color=BLUE, linewidth=1.5)
            if j == 1:
                ax.axvspan(sick["start_day"], 59, color=ORANGE, alpha=0.12, linewidth=0)
            ax.set_ylabel(lab, fontsize=9)
            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)
            ax.grid(alpha=0.25, linewidth=0.6)
        axes[0, j].set_title(f"SIMULATED patient (ecg_id {pid}): {title}", loc="left", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("Day")
    fig.tight_layout()
    fig.savefig(OUT / "example_patients.png", dpi=140)
    plt.close(fig)


def main():
    recipe = json.load(open("outputs/recipe/recipe.json", encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    all_sim, all_eps = [], []
    for split in ["val", "test"]:
        meta = pd.read_csv(PROC / f"meta_{split}.csv")
        sim, eps = simulate_split(meta, recipe, rng, split)
        sim.to_csv(PROC / f"wearables_{split}.csv", index=False)
        eps.to_csv(PROC / f"episodes_{split}.csv", index=False)
        all_sim.append(sim)
        all_eps.append(eps)
        n_ep = int(eps["episode"].sum())
        log(f"{split}: {len(eps)} patients x {N_DAYS} days | with episode {n_ep} "
            f"({100 * n_ep / len(eps):.1f}%) | severity counts "
            f"{eps.loc[eps['episode'] == 1, 'severity'].value_counts().sort_index().to_dict()}")

    # Independence check: episode assignment must not depend on the ECG label
    eps = pd.concat(all_eps)
    labels = pd.concat([pd.read_csv(PROC / f"meta_{s}.csv")[["ecg_id", "label"]] for s in ["val", "test"]])
    tab = eps.merge(labels, on="ecg_id").groupby("label")["episode"].mean()
    log(f"\nEpisode share by ECG label (should be about equal): normal {tab.get(0, np.nan):.3f} | "
        f"abnormal {tab.get(1, np.nan):.3f}")

    sim_val = all_sim[0]
    log("\nSimulated levels (validation, day 0-34, before any episode): "
        f"resting HR median {sim_val[sim_val.day < 35]['resting_hr'].median():.1f} bpm | "
        f"HRV median {sim_val[sim_val.day < 35]['hrv_rmssd'].median():.1f} ms | "
        f"sleep median {sim_val[sim_val.day < 35]['sleep_hours'].median():.2f} h | "
        f"steps median {sim_val[sim_val.day < 35]['steps'].median():.0f}")

    pm = pd.read_csv(PROC / "pmdata_daily.csv", parse_dates=["date"])
    realism_check(sim_val, pm)
    example_plot(sim_val, all_eps[0])

    log(f"\nSaved wearables_val/test.csv, episodes_val/test.csv and two PNGs in {OUT}")
    with open(OUT / "simulation_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
