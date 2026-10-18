"""
fit_recipe.py: Stage 2 of the wearable stream (DECISIONS.md D-13, D-14, D-15).

Builds the "recipe" the simulator will follow. Every number written to
recipe.json carries a "source" field saying where it came from:
  - "PMData (computed)"     measured here from the real PMData days
  - "MMASH (computed)"      measured here from the real MMASH nights
  - "<paper>, <section>"    taken from a published paper (verified 05 Oct 2026)
  - "derived from ..."      a simple calculation from a cited number
  - "UNSOURCED"             a gap that must be filled before simulation

THE MODEL FOR ONE PERSON AND ONE MEASURE (resting HR, sleep, steps)

    value on day t = personal level
                   + weekday effect (Mon ... Sun)
                   + wobble_t

    wobble_t = phi * wobble_(t-1) + fresh noise
      phi   "memory": how much of yesterday's deviation carries into today
            (0 = none, close to 1 = deviations last many days)
      sd    typical size of the wobble

Steps are modelled on a log scale, so effects are multiplicative
(a weekend is "20% fewer steps", not "2,000 fewer steps").

WHAT COMES FROM WHERE (D-15, option B)
  personal level     population studies (resting HR, steps); PMData for sleep;
                     MMASH for HRV (until age-adjusted sources are verified)
  weekday, phi, sd   PMData only: the real day-to-day behaviour

Inputs:  data/processed/pmdata_daily.csv, data/processed/mmash_hrv.csv
Outputs: outputs/recipe/recipe.json
         outputs/recipe/per_person_dynamics.csv
         outputs/recipe/recipe_summary.txt
Run:     python fit_recipe.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")
OUT = Path("outputs/recipe")
OUT.mkdir(parents=True, exist_ok=True)

MIN_DAYS = 60          # a person needs at least 60 valid days to contribute dynamics
MEASURES = {           # column -> (scale, unit)
    "resting_hr": ("linear", "bpm"),
    "sleep_hours": ("linear", "hours"),
    "steps": ("log", "steps/day"),
}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

log_lines = []


def log(text=""):
    print(text)
    log_lines.append(str(text))


def src(value, source, note=None):
    d = {"value": value, "source": source}
    if note:
        d["note"] = note
    return d


# ---------------------------------------------------------------------------
# Dynamics from PMData
# ---------------------------------------------------------------------------
def person_dynamics(series, scale):
    """series: values indexed by date for one person and one measure."""
    s = series.dropna()
    if scale == "log":
        s = np.log(s[s > 0])
    level = s.mean()
    dev = s - level
    weekday = dev.groupby(dev.index.dayofweek).mean()          # 0 = Monday
    weekday = weekday.reindex(range(7)).fillna(0.0)
    resid = dev - weekday.reindex(dev.index.dayofweek).to_numpy()

    # lag-1 autocorrelation using only true consecutive calendar days
    r = resid.copy()
    gap = r.index.to_series().diff().dt.days.to_numpy()
    pairs = gap[1:] == 1
    x, y = r.to_numpy()[:-1][pairs], r.to_numpy()[1:][pairs]
    phi = np.corrcoef(x, y)[0, 1] if len(x) > 10 else np.nan
    return {"n_days": len(s), "level": level, "sd": resid.std(ddof=1),
            "phi": phi, "n_pairs": int(pairs.sum()),
            **{f"wd_{DAYS[i]}": weekday[i] for i in range(7)}}


def fit_dynamics(daily):
    rows = []
    for col, (scale, _) in MEASURES.items():
        for pid, d in daily.groupby("pid"):
            s = d.set_index("date")[col]
            n = s.dropna().shape[0]
            if n < MIN_DAYS:
                rows.append({"measure": col, "pid": pid, "n_days": n, "used": False})
                continue
            rows.append({"measure": col, "pid": pid, "used": True, **person_dynamics(s, scale)})
    return pd.DataFrame(rows)


def pooled(per, col):
    p = per[(per["measure"] == col) & per["used"]]
    sd = p["sd"].to_numpy()
    return {
        "n_people": int(len(p)),
        "people": list(p["pid"]),
        "phi_median": float(np.median(p["phi"])),
        "phi_range": [float(p["phi"].min()), float(p["phi"].max())],
        "sd_median": float(np.median(sd)),
        # how much people differ in wobble size: SD of log(sd) across people
        "sd_log_spread": float(np.std(np.log(sd), ddof=1)),
        "sd_range": [float(sd.min()), float(sd.max())],
        "weekday": {d: float(p[f"wd_{d}"].mean()) for d in DAYS},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    daily = pd.read_csv(PROC / "pmdata_daily.csv", parse_dates=["date"])
    hrv = pd.read_csv(PROC / "mmash_hrv.csv")

    per = fit_dynamics(daily)
    per.to_csv(OUT / "per_person_dynamics.csv", index=False)

    log("=" * 72)
    log(f"DAY-TO-DAY DYNAMICS FROM PMDATA (people with at least {MIN_DAYS} valid days)")
    log("=" * 72)
    dyn = {}
    for col, (scale, unit) in MEASURES.items():
        p = per[per["measure"] == col]
        dropped = p[~p["used"]]
        dyn[col] = pooled(per, col)
        g = dyn[col]
        u = "log units (x100 = approx. %)" if scale == "log" else unit
        log(f"\n{col}  [{scale} scale]")
        log(f"  people used: {g['n_people']}  {g['people']}")
        log(f"  excluded (< {MIN_DAYS} days): "
            + ", ".join(f"{r.pid} ({int(r.n_days)})" for r in dropped.itertuples()) if len(dropped) else "  excluded: none")
        log(f"  memory phi:  median {g['phi_median']:.2f}  (range {g['phi_range'][0]:.2f} to {g['phi_range'][1]:.2f})")
        log(f"  wobble sd:   median {g['sd_median']:.3f} {u}  (range {g['sd_range'][0]:.3f} to {g['sd_range'][1]:.3f})")
        log("  weekday effect: " + "  ".join(f"{d} {v:+.3f}" for d, v in g["weekday"].items()))

    # ---------------------------------------------------------------------
    # Personal levels (between-person distributions)
    # ---------------------------------------------------------------------
    # Resting HR, Quer et al. 2020: 95% of women 53-82 bpm, 95% of men 50-80 bpm.
    # Assuming a normal distribution, mean = midpoint and SD = width / 3.92.
    rhr = {sex: {"mean": (lo + hi) / 2, "sd": (hi - lo) / 3.92}
           for sex, (lo, hi) in {"female": (53, 82), "male": (50, 80)}.items()}

    # Steps, Althoff et al. 2017: India mean 4,297; between-user SD 2,684 (worldwide).
    # Converted to a log-normal with that mean and SD.
    m, s = 4297.0, 2684.0
    sig2 = np.log(1 + (s / m) ** 2)
    steps_mu, steps_sigma = np.log(m) - sig2 / 2, np.sqrt(sig2)

    sleep_levels = per[(per["measure"] == "sleep_hours") & per["used"]]["level"]

    valid_hrv = hrv["rmssd_ms"].dropna()
    log_hrv = np.log(valid_hrv)

    levels = {
        "resting_hr_bpm": {
            "distribution": "normal, by sex",
            "female": {"mean": src(rhr["female"]["mean"], "derived from Quer et al. 2020 (PLOS ONE), Results: 95% of women 53-82 bpm"),
                       "sd": src(round(rhr["female"]["sd"], 2), "derived: (82-53)/3.92, assumes a normal distribution")},
            "male": {"mean": src(rhr["male"]["mean"], "derived from Quer et al. 2020 (PLOS ONE), Results: 95% of men 50-80 bpm"),
                     "sd": src(round(rhr["male"]["sd"], 2), "derived: (80-50)/3.92, assumes a normal distribution")},
            "cross_check": src("65.5 +/- 7.7", "Quer et al. 2020, Results: mean RHR over all individuals and days"),
            "age_effect": src(None, "not applied", "Quer et al. 2020 report a rise to about age 50 then a decline (Fig. 3A), but give no numeric values"),
        },
        "steps_per_day": {
            "distribution": "log-normal",
            "mu_log": src(round(float(steps_mu), 4), "derived from mean 4,297 and SD 2,684"),
            "sigma_log": src(round(float(steps_sigma), 4), "derived from mean 4,297 and SD 2,684"),
            "mean": src(4297, "Althoff et al. 2017 (Nature), Supplementary Table 1: India"),
            "sd": src(2684, "Althoff et al. 2017 (Nature), Methods: SD across all users worldwide"),
            "note": "smartphone-measured; phones typically record fewer steps than wrist trackers",
            "age_effect": src(None, "not applied", "Althoff et al. 2017 report lower activity with age (Extended Data Fig. 1) without values in the text retrieved"),
        },
        "sleep_hours": {
            "distribution": "normal",
            "mean": src(round(float(sleep_levels.mean()), 3), f"PMData (computed): mean of {len(sleep_levels)} personal mean sleep durations"),
            "sd": src(round(float(sleep_levels.std(ddof=1)), 3), "PMData (computed): SD across those people"),
            "note": "placeholder until Ohayon et al. 2004 (Sleep) age norms are verified",
        },
        "hrv_rmssd_ms": {
            "distribution": "log-normal",
            "mu_log": src(round(float(log_hrv.mean()), 4), f"MMASH (computed): mean log night RMSSD of {len(valid_hrv)} users"),
            "sigma_log": src(round(float(log_hrv.std(ddof=1)), 4), "MMASH (computed): SD of log night RMSSD"),
            "median_ms": src(round(float(valid_hrv.median()), 1), "MMASH (computed)"),
            "note": "young healthy men only; age adjustment awaits verification of Umetani et al. 1998 (JACC)",
        },
    }

    def dyn_entry(col, unit):
        g = dyn[col]
        source = f"PMData (computed): {g['n_people']} people with >= {MIN_DAYS} valid days"
        return {"unit": unit,
                "phi": src(round(g["phi_median"], 3), source + ", median lag-1 autocorrelation"),
                "sd": src(round(g["sd_median"], 4), source + ", median residual SD"),
                "sd_log_spread": src(round(g["sd_log_spread"], 3), source + ", SD of log(residual SD) across people"),
                "weekday": src({k: round(v, 4) for k, v in g["weekday"].items()}, source + ", mean weekday deviation")}

    dynamics = {
        "resting_hr_bpm": dyn_entry("resting_hr", "bpm"),
        "sleep_hours": dyn_entry("sleep_hours", "hours"),
        "steps_per_day": dyn_entry("steps", "log steps"),
        "hrv_rmssd_ms": {
            "unit": "log ms",
            # Hannon et al. 2025, Sensors 25(14):4415, Section 4 (Discussion): CV of RMSSD over
            # 14 days per participant: average 0.37 (SD 0.16), range 0.14-0.71; 41 healthy adults,
            # 5-minute morning recordings, Polar H10 chest strap.
            # A coefficient of variation is converted to a log-scale SD with sqrt(ln(1 + CV^2)).
            "sd": src(round(float(np.sqrt(np.log(1 + 0.37 ** 2))), 4),
                      "derived from Hannon et al. 2025 (Sensors 25(14):4415), Section 4: average 14-day CV of RMSSD 0.37",
                      "morning 5-min readings, while our HRV levels are night-time (MMASH); a known mismatch"),
            "sd_log_spread": src(None, "not estimated",
                                 "Hannon et al. 2025 report only the CV mean (0.37), SD (0.16) and range (0.14-0.71); "
                                 "the simulator uses the resting HR spread as a stand-in"),
            "phi": src(round(dyn["resting_hr"]["phi_median"], 3),
                       "ASSUMPTION (conservative): borrowed from PMData resting HR memory",
                       "no source found for night-to-night HRV memory; a high value makes natural drifts "
                       "longer and so makes detection harder, not easier"),
            "weekday": src({d: 0.0 for d in DAYS}, "ASSUMPTION: no weekday effect for HRV"),
        },
    }

    recipe = {
        "created": "by fit_recipe.py",
        "decisions": ["D-13", "D-14", "D-14a", "D-15"],
        "levels": levels,
        "dynamics": dynamics,
        "licences": {"PMData": "CC BY-NC 4.0 (only these summary numbers are used; raw data not redistributed)",
                     "MMASH": "ODbL v1.0",
                     "Quer et al. 2020": "CC BY (article)",
                     "Althoff et al. 2017": "Nature article; numbers cited, not reproduced"},
        "open_gaps": [
            "HRV day-to-day memory (phi): ASSUMED equal to resting HR memory; no source found",
            "HRV age adjustment: awaiting Umetani et al. 1998 full text",
            "Sleep level by age: awaiting Ohayon et al. 2004 values",
            "Age effects on resting HR and steps: sources give direction only",
        ],
    }
    with open(OUT / "recipe.json", "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2)

    log("\n" + "=" * 72)
    log("PERSONAL LEVELS (between-person)")
    log("=" * 72)
    log(f"Resting HR  women mean {rhr['female']['mean']:.1f} sd {rhr['female']['sd']:.2f} | "
        f"men mean {rhr['male']['mean']:.1f} sd {rhr['male']['sd']:.2f} bpm  (Quer 2020, derived)")
    log(f"Steps       log-normal, mean 4,297 sd 2,684  (Althoff 2017, India)")
    log(f"Sleep       mean {sleep_levels.mean():.2f} h, sd {sleep_levels.std(ddof=1):.2f} h across "
        f"{len(sleep_levels)} PMData people (placeholder)")
    log(f"HRV RMSSD   median {valid_hrv.median():.1f} ms, log-sd {log_hrv.std(ddof=1):.3f} across "
        f"{len(valid_hrv)} MMASH users")
    log("\nOPEN GAPS: " + "; ".join(recipe["open_gaps"]))
    log(f"\nSaved {OUT / 'recipe.json'} and {OUT / 'per_person_dynamics.csv'}")

    with open(OUT / "recipe_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
