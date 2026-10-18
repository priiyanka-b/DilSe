"""
explore_wearables.py: Stage 1 of the wearable stream (DECISIONS.md D-13, D-14).

Reads the two real wearable datasets and turns them into clean tables.
It does NOT simulate anything yet; it only answers "what real data do we have?"

PMData (data/raw/pmdata/pmdata/p01 ... p16/fitbit/)    licence CC BY-NC 4.0
  For every participant and every day:
    resting_hr   from resting_heart_rate.json   (zero values = no reading, dropped)
    sleep_hours  from sleep.json, main sleep only (exact duplicate records removed)
    steps        from steps.json, minute counts summed per calendar day
  Days with fewer than MIN_STEP_MINUTES minutes of step data are marked missing,
  because a watch worn for 2 hours would look like a very lazy day.

MMASH (data/raw/mmash/DataPaper/user_1 ... user_22/)   licence ODbL v1.0
  For every user: night-time HRV (RMSSD) and mean heart rate during sleep,
  computed from the beat-to-beat intervals in RR.csv, after artifact cleaning:
    1. keep beat intervals between 0.3 s and 2.0 s (30-200 bpm)
    2. drop a beat if it differs from the previous beat by more than 20%
  RMSSD = root mean square of differences between successive beats, in ms.
  It is the standard short-term HRV measure; higher usually means more
  parasympathetic ("rest and digest") activity.

Run from C:\\Users\\priya\\dt-cardio:
    python explore_wearables.py

Outputs:
  data/processed/pmdata_daily.csv            one row per participant-day
  data/processed/mmash_hrv.csv               one row per MMASH user
  outputs/wearables/summary.txt              every number printed below
  outputs/wearables/pmdata_resting_hr.png    small multiples, one panel per person
  outputs/wearables/pmdata_sleep.png
  outputs/wearables/pmdata_steps.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Mean of empty slice")

PM_DIR = Path("data/raw/pmdata/pmdata")
MM_DIR = Path("data/raw/mmash/DataPaper")
PROC = Path("data/processed")
OUT = Path("outputs/wearables")
PROC.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

MIN_STEP_MINUTES = 600      # a valid steps day needs at least 10 h of minute records
RR_MIN, RR_MAX = 0.3, 2.0   # seconds between beats (200 to 30 bpm)
RR_JUMP = 0.20              # max relative change from one beat to the next
MAX_PCT_REMOVED = 20.0      # a night with more than 20% artifact beats is too noisy to trust
LINE = "#2a78d6"            # single series colour

log_lines = []


def log(text=""):
    print(text)
    log_lines.append(str(text))


# ---------------------------------------------------------------------------
# PMData
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pm_resting_hr(fitbit):
    path = fitbit / "resting_heart_rate.json"
    if not path.exists():
        return pd.Series(dtype=float, name="resting_hr")
    rows = [(r["dateTime"][:10], r["value"]["value"]) for r in load_json(path)]
    s = pd.DataFrame(rows, columns=["date", "resting_hr"]).drop_duplicates("date")
    s = s[s["resting_hr"] > 0]
    return s.set_index("date")["resting_hr"]


def pm_sleep(fitbit):
    path = fitbit / "sleep.json"
    if not path.exists():
        return pd.Series(dtype=float, name="sleep_hours"), 0
    recs = load_json(path)
    n_raw = len(recs)
    df = pd.DataFrame([{"logId": r["logId"], "date": r["dateOfSleep"],
                        "main": r.get("mainSleep", True),
                        "minutes": r["minutesAsleep"]} for r in recs])
    df = df.drop_duplicates("logId")          # the export repeats some records
    n_dupes = n_raw - len(df)
    df = df[df["main"]]
    s = df.groupby("date")["minutes"].sum() / 60.0
    return s.rename("sleep_hours"), n_dupes


def pm_steps(fitbit):
    path = fitbit / "steps.json"
    if not path.exists():
        return pd.Series(dtype=float, name="steps")
    df = pd.DataFrame(load_json(path))
    df["date"] = df["dateTime"].str[:10]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    g = df.groupby("date")["value"].agg(["sum", "count"])
    g.loc[g["count"] < MIN_STEP_MINUTES, "sum"] = np.nan
    return g["sum"].rename("steps")


def load_pmdata():
    overview = pd.read_excel(PM_DIR / "participant-overview.xlsx", header=1)
    overview = overview.rename(columns={"Participant ID": "pid", "Age": "age", "Gender": "sex"})
    overview["sex"] = overview["sex"].astype(str).str.strip()
    overview = overview[["pid", "age", "sex"]].dropna(subset=["pid"])

    frames, dupes = [], {}
    for pid in sorted(p.name for p in PM_DIR.glob("p*") if p.is_dir()):
        fitbit = PM_DIR / pid / "fitbit"
        sleep, n_d = pm_sleep(fitbit)
        dupes[pid] = n_d
        day = pd.concat([pm_resting_hr(fitbit), sleep, pm_steps(fitbit)], axis=1)
        day.index.name = "date"
        day = day.reset_index()
        day["pid"] = pid
        frames.append(day)
    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.merge(overview, on="pid", how="left")
    daily = daily.sort_values(["pid", "date"])[["pid", "age", "sex", "date", "resting_hr", "sleep_hours", "steps"]]
    return daily, dupes


def day_to_day_sd(series):
    """SD of the change between consecutive calendar days (only true neighbours)."""
    s = series.dropna()
    gaps = s.index.to_series().diff().dt.days
    d = s.diff()[gaps == 1]
    return d.std() if len(d) > 2 else np.nan


def small_multiples(daily, col, ylabel, fname):
    pids = sorted(daily["pid"].unique())
    pids = [p for p in pids if daily.loc[daily["pid"] == p, col].notna().any()]
    n = len(pids)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 2.4 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, pid in zip(axes, pids):
        d = daily[daily["pid"] == pid].set_index("date")[col]
        d = d.asfreq("D")                       # missing days become gaps, not straight lines
        ax.plot(d.index, d.values, color=LINE, linewidth=1.2)
        loc = mdates.AutoDateLocator(maxticks=4)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        ax.set_title(pid, loc="left", fontsize=10)
        ax.grid(alpha=0.25, linewidth=0.6)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.supylabel(ylabel)
    fig.suptitle(f"PMData: {ylabel}, one panel per participant (real data)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# MMASH
# ---------------------------------------------------------------------------
def clock_seconds(day, time_str):
    """MMASH stores 'day' (1 or 2) plus 'HH:MM[:SS]'. Convert to seconds from day 1, 00:00."""
    t = pd.to_timedelta(time_str.where(time_str.str.count(":") == 2, time_str + ":00"))
    return (day.astype(int) - 1) * 86400 + t.dt.total_seconds()


def mmash_user(udir):
    info = pd.read_csv(udir / "user_info.csv", index_col=0).iloc[0]
    rr = pd.read_csv(udir / "RR.csv", index_col=0)
    sleep = pd.read_csv(udir / "sleep.csv", index_col=0)
    row = {"user": udir.name, "age": info.get("Age"), "sex": info.get("Gender"),
           "beats_raw": len(rr)}
    if sleep.empty:
        return {**row, "note": "no sleep record"}

    # Users 8 and 9 store the second recording day as "-29" instead of 2
    # (their day 1 runs ~09:50-23:59, the "-29" rows run 00:00-09:20).
    # Any day number below 1 is therefore treated as day 2.
    n_bad_day = int((rr["day"] < 1).sum())
    rr["day"] = rr["day"].where(rr["day"] >= 1, 2)
    rr["t"] = clock_seconds(rr["day"], rr["time"].astype(str))
    starts = clock_seconds(sleep["In Bed Date"], sleep["In Bed Time"].astype(str))
    ends = clock_seconds(sleep["Out Bed Date"], sleep["Out Bed Time"].astype(str))
    # Some MMASH files label the out-of-bed time with the SAME day as going to bed
    # (e.g. in bed day 1 22:29, out of bed "day 1" 05:52). An end before the start
    # can only mean the next morning, so add one day.
    ends = ends.where(ends > starts, ends + 86400)
    start, end = starts.min(), ends.max()
    night = rr[(rr["t"] >= start) & (rr["t"] <= end)]["ibi_s"].to_numpy()
    if len(night) < 3:
        return {**row, "note": "no beats recorded during the sleep period"}

    ok = (night >= RR_MIN) & (night <= RR_MAX)
    ibi = night[ok]
    if len(ibi) > 1:
        jump = np.abs(np.diff(ibi)) / ibi[:-1]
        ibi = ibi[np.concatenate([[True], jump <= RR_JUMP])]
    pct_removed = 100 * (1 - len(ibi) / len(night))
    if n_bad_day:
        row["note"] = f"{n_bad_day} beats had day coded below 1, treated as day 2"
    row.update({"night_beats_raw": len(night),
                "night_beats_clean": len(ibi),
                "pct_removed": pct_removed,
                "sleep_hr_bpm": 60.0 / ibi.mean() if len(ibi) else np.nan,
                "rmssd_ms": 1000 * np.sqrt(np.mean(np.diff(ibi) ** 2)) if len(ibi) > 2 else np.nan})
    if pct_removed > MAX_PCT_REMOVED:
        row["note"] = f"excluded: {pct_removed:.0f}% of beats were artifacts (limit {MAX_PCT_REMOVED:.0f}%)"
        row["rmssd_ms"] = np.nan
    if row.get("age") == 0:
        row["age"] = np.nan
        row["note"] = (row.get("note", "") + " age recorded as 0 in user_info.csv, set to unknown").strip()
    return row


def main():
    log("=" * 72)
    log("PMDATA (CC BY-NC 4.0): real daily wearable data")
    log("=" * 72)
    daily, dupes = load_pmdata()
    daily.to_csv(PROC / "pmdata_daily.csv", index=False)

    rows = []
    for pid, d in daily.groupby("pid"):
        d = d.set_index("date")
        rows.append({"pid": pid, "age": d["age"].iloc[0], "sex": d["sex"].iloc[0],
                     "days_span": (d.index.max() - d.index.min()).days + 1,
                     "rhr_days": d["resting_hr"].notna().sum(),
                     "sleep_days": d["sleep_hours"].notna().sum(),
                     "steps_days": d["steps"].notna().sum(),
                     "rhr_median": d["resting_hr"].median(),
                     "rhr_dd_sd": day_to_day_sd(d["resting_hr"]),
                     "sleep_median_h": d["sleep_hours"].median(),
                     "steps_median": d["steps"].median(),
                     "sleep_dupes_removed": dupes.get(pid, 0)})
    per = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        log(per.to_string(index=False))
    log(f"\nTotal valid days: resting HR {int(per['rhr_days'].sum())} | "
        f"sleep {int(per['sleep_days'].sum())} | steps {int(per['steps_days'].sum())}")
    log(f"Participants with any resting HR: {(per['rhr_days'] > 0).sum()} of {len(per)}")
    log("rhr_dd_sd = SD of the change between consecutive days (bpm): the 'day-to-day wobble'.")

    small_multiples(daily, "resting_hr", "Resting heart rate (bpm)", "pmdata_resting_hr.png")
    small_multiples(daily, "sleep_hours", "Main sleep (hours)", "pmdata_sleep.png")
    small_multiples(daily, "steps", "Steps per day", "pmdata_steps.png")

    log("\n" + "=" * 72)
    log("MMASH (ODbL v1.0): night-time HRV from beat-to-beat intervals")
    log("=" * 72)
    users = sorted((p for p in MM_DIR.glob("user_*") if p.is_dir()),
                   key=lambda p: int(p.name.split("_")[1]))
    hrv = []
    for u in users:
        try:
            hrv.append(mmash_user(u))
        except Exception as e:                    # report, don't crash on one bad user
            hrv.append({"user": u.name, "note": f"error: {e}"})
    hrv = pd.DataFrame(hrv)
    hrv.to_csv(PROC / "mmash_hrv.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        log(hrv.to_string(index=False))
    valid = hrv["rmssd_ms"].dropna() if "rmssd_ms" in hrv else pd.Series(dtype=float)
    if len(valid):
        log(f"\nUsers with a valid night RMSSD: {len(valid)} of {len(hrv)} | "
            f"median {valid.median():.1f} ms, range {valid.min():.1f}-{valid.max():.1f} ms")

    log("\nSaved data/processed/pmdata_daily.csv, data/processed/mmash_hrv.csv, "
        "and three PNGs in outputs/wearables/")
    with open(OUT / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
