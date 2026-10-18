"""
dashboard.py: DilSe, the doctor-facing app (DECISIONS.md D-19, D-20, D-22, D-24, D-25).

Three screens
  1. WORKLIST      every monitored patient today, most urgent first, with what changed,
                   what is new since yesterday and a suggested next step.
  2. PATIENT       one patient: a one-line summary, the suggested next step, the twin's
                   profile of this person, today's readings against their usual values,
                   trends, status history, the 12-lead ECG, review notes and a PDF summary.
  3. HOW IT WORKS  how the status is decided, test results, limits.

Everything shown is computed live from files the pipeline already wrote; nothing is
re-tuned here. All settings are read from the locked config files.
  ECG stream      outputs/final_test/test_predictions.csv   REAL ECGs, 4-CNN ensemble (D-12)
  wearable stream data/processed/wearables_test.csv          SYNTHETIC (D-13 to D-16)
  monitor         monitor.py (days 35-59, D-17) and learning_twin.py (days 0-34, D-24)
  twin settings   outputs/twin/twin_config.json, outputs/learning/learning_config.json
  12-lead ECG     data/processed/X_test.npy if present, else data/demo/ecg_demo.npz (99 patients)
Only TEST-split patients are shown: the model never saw them during development.

The "Demo controls" replay the 60 simulated days. A real clinic would only ever see today.

Run (from the dt-cardio folder):  streamlit run dashboard.py
"""

import datetime as dt
import io
import json
import textwrap
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from learning_twin import (LEARN_DAYS, learning_score, learning_state, load_recipe, prior_for,
                           tier_learning_flags)
from monitor import LOG_SCALE, MEASURES, ROLL_GAP, ROLL_LEN, SMOOTH, daily_scores, flags_from, load

PROC = Path("data/processed")
FINAL = Path("outputs/final_test")
TWIN_CFG = Path("outputs/twin/twin_config.json")
LEARN_CFG = Path("outputs/learning/learning_config.json")
DEMO_ECG = Path("data/demo/ecg_demo.npz")
N_DAYS = 60
DAY0 = dt.date(2026, 8, 17)          # display date of simulated day 0 (day 59 = 15 Oct 2026)

STATES = ["STABLE", "WATCH", "ESCALATE"]
LABEL = {"STABLE": "Stable", "WATCH": "Concern", "ESCALATE": "Critical"}  # display names (internal codes unchanged, D-25)
COLOR = {"STABLE": "#3d7a5a", "WATCH": "#c98a00", "ESCALATE": "#c0392b"}
TINT = {"STABLE": "#e8f2ec", "WATCH": "#fbf1dc", "ESCALATE": "#f8e3e0"}
DOT = {"STABLE": "🟢", "WATCH": "🟠", "ESCALATE": "🔴"}
MEANING = {
    "STABLE": "Low-risk ECG and readings within the patient's usual range.",
    "WATCH": "High-risk ECG, or a lasting change in a low-risk patient's readings.",
    "ESCALATE": "High-risk ECG and a lasting change away from the patient's usual range.",
}
NEXT = {   # triage suggestions only (no diagnosis, no treatment advice)
    "ESCALATE": ("Contact the patient within 24 hours", "Consider bringing the clinic review forward.", "Contact ≤ 24 h"),
    "WATCH_HIGH": ("Review the trend again in 3 days", "High-risk ECG; readings are currently within the usual range.",
                   "Review in 3 days"),
    "WATCH_LOW": ("Check in with the patient within 3 days", "Low-risk ECG, but readings have stayed away from the usual "
                                                             "range.", "Check in ≤ 3 days"),
    "STABLE": ("Routine follow-up", "No action needed today.", "Routine"),
}
NICE = {"resting_hr": "Resting heart rate", "hrv_rmssd": "Heart-rate variability",
        "sleep_hours": "Sleep", "steps": "Steps"}
SHORT = {"resting_hr": "resting HR", "hrv_rmssd": "HRV", "sleep_hours": "sleep", "steps": "steps"}
UNIT = {"resting_hr": "bpm", "hrv_rmssd": "ms", "sleep_hours": "h", "steps": "steps/day"}
BAD_WAY = {"resting_hr": "above", "hrv_rmssd": "below", "sleep_hours": "below", "steps": "below"}
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]  # PTB-XL lead order

st.set_page_config(page_title="DilSe", page_icon="🫀", layout="wide")

st.markdown("""
<style>
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.ct-bar {display:flex; justify-content:space-between; align-items:center; background:#12324a; color:#f3f6f8;
         border-radius:12px; padding:14px 22px; margin-bottom:18px;}
.ct-bar .brand {font-size:1.35rem; font-weight:700; letter-spacing:.2px;}
.ct-bar .brand span {font-weight:400; color:#a9c3d3; font-size:.95rem; margin-left:10px;}
.ct-bar .when {text-align:right; font-size:.92rem; color:#d7e3ea;}
.ct-bar .when b {color:#ffffff;}
.ct-kpi {border-radius:12px; padding:14px 18px; border:1px solid #e3e8ec; background:#ffffff; margin-bottom:12px;}
.ct-kpi .lab {font-size:.85rem; font-weight:600; letter-spacing:.4px; text-transform:uppercase;}
.ct-kpi .num {font-size:2.1rem; font-weight:700; line-height:1.15; color:#1d2733;}
.ct-kpi .sub {font-size:.85rem; color:#5b6773;}
.ct-pill {display:inline-block; padding:3px 12px; border-radius:999px; color:#fff; font-weight:600;
          font-size:.9rem; vertical-align:middle;}
.ct-new {display:inline-block; padding:2px 9px; border-radius:999px; background:#12324a; color:#fff;
         font-size:.75rem; font-weight:700; letter-spacing:.5px; vertical-align:middle; margin-left:8px;}
.ct-summary {font-size:1.15rem; line-height:1.55; color:#1d2733; margin:6px 0 10px 0;}
.ct-next {border-radius:12px; padding:14px 18px; margin:8px 0 14px 0;}
.ct-next .t {font-size:1.1rem; font-weight:700;}
.ct-next .d {font-size:.95rem; color:#33414d;}
.ct-muted {color:#6a7682; font-size:.88rem;}
.ct-event {font-size:.95rem; padding:4px 0; border-bottom:1px solid #eef1f3;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------------
# Data: loaded once and cached (logic unchanged from the locked pipeline)
# ----------------------------------------------------------------------------------
@st.cache_data
def load_everything():
    cfg = json.load(open(TWIN_CFG))
    pred = pd.read_csv(FINAL / "test_predictions.csv").set_index("ecg_id")
    X, eps = load("test")                      # (patients, 60 days, 4 measures); HRV/steps on log scale
    pred = pred.loc[eps.index]
    score, z = daily_scores(X, "rolling")      # locked monitor, days 35-59

    high = pred["p_ecg"].to_numpy() >= cfg["risk_threshold_p_ecg"]
    flags = np.zeros(score.shape, bool)
    for mask, s in [(~high, cfg["low_risk"]), (high, cfg["high_risk"])]:
        flags[mask] = flags_from(score[mask], s["threshold"], s["persistence"])

    # learning twin (D-24): days 0-34. Days 35-59 above are the locked monitor, unchanged.
    learn_cfg = json.load(open(LEARN_CFG)) if LEARN_CFG.exists() else None
    learn = None
    if learn_cfg is not None:
        recipe = load_recipe()
        pri = tuple(np.stack([prior_for(recipe, sx)[k] for sx in eps["sex"]]) for k in range(4))
        learn = learning_state(X, pri)
        l_score = learning_score(learn["z"])
        early = np.arange(N_DAYS)[None, :] < LEARN_DAYS
        score = np.where(early, l_score, score)
        z = np.where(early[:, :, None], learn["z"], z)
        if learn_cfg.get("alarms_enabled"):
            flags = flags | tier_learning_flags(l_score, high, learn_cfg)
    state = np.where(flags & high[:, None], 2, np.where(flags | high[:, None], 1, 0))

    z_s = np.stack([pd.DataFrame(z[:, :, m].T).rolling(SMOOTH, min_periods=SMOOTH).mean().to_numpy().T
                    for m in range(len(MEASURES))], axis=2)
    normal = np.full(X.shape, np.nan)          # the patient's usual value, as the monitor sees it
    lo = np.full(X.shape, np.nan)
    hi = np.full(X.shape, np.nan)
    for t in range(ROLL_LEN + ROLL_GAP, N_DAYS):
        b = X[:, t - ROLL_GAP - ROLL_LEN:t - ROLL_GAP, :]
        med = np.median(b, axis=1)
        sp = 1.4826 * np.median(np.abs(b - med[:, None, :]), axis=1)
        normal[:, t], lo[:, t], hi[:, t] = med, med - sp, med + sp
    learned = np.ones(X.shape)
    if learn is not None:
        normal[:, :LEARN_DAYS] = learn["level"][:, :LEARN_DAYS]
        lo[:, :LEARN_DAYS] = (learn["level"] - learn["pred_sd"])[:, :LEARN_DAYS]
        hi[:, :LEARN_DAYS] = (learn["level"] + learn["pred_sd"])[:, :LEARN_DAYS]
        learned[:, :LEARN_DAYS] = learn["learned"][:, :LEARN_DAYS]
    today_val = np.stack([pd.DataFrame(X[:, :, m].T).rolling(SMOOTH, min_periods=1).mean().to_numpy().T
                          for m in range(len(MEASURES))], axis=2)
    vals = X.copy()
    for arr in (normal, lo, hi, today_val, vals):
        arr[:, :, LOG_SCALE] = np.exp(arr[:, :, LOG_SCALE])

    # consecutive flagged days ending on each day
    run = np.zeros(flags.shape, int)
    for t in range(N_DAYS):
        run[:, t] = np.where(flags[:, t], (run[:, t - 1] if t else 0) + 1, 0)
    return dict(cfg=cfg, learn_cfg=learn_cfg, pred=pred, eps=eps, score=score, z_s=z_s, high=high,
                flags=flags, state=state, normal=normal, lo=lo, hi=hi, learned=learned,
                today_val=today_val, vals=vals, run=run)


@st.cache_resource
def ecg_source():
    """Full PTB-XL test array if present (local), else the bundled demo subset (hosted app)."""
    meta = pd.read_csv(PROC / "meta_test.csv")
    meta_row = pd.Series(np.arange(len(meta)), index=meta["ecg_id"])
    full, sal_path = PROC / "X_test.npy", Path("outputs/saliency/saliency_test.npy")
    if full.exists():
        X = np.load(full, mmap_mode="r")
        S = np.load(sal_path, mmap_mode="r") if sal_path.exists() else None
        return {"get": lambda pid: (np.asarray(X[meta_row[pid]], np.float32)[:, ::2],
                                    None if S is None else np.asarray(S[meta_row[pid]], np.float32)),
                "has": lambda pid: pid in meta_row.index}
    if DEMO_ECG.exists():
        d = np.load(DEMO_ECG)
        idx = {int(p): k for k, p in enumerate(d["ecg_id"])}
        E, S = d["ecg"], d["saliency"]
        return {"get": lambda pid: (E[idx[pid]].astype(np.float32), S[idx[pid]].astype(np.float32)),
                "has": lambda pid: pid in idx}
    return None


D = load_everything()
cfg, learn_cfg, pred, eps = D["cfg"], D["learn_cfg"], D["pred"], D["eps"]
state, flags, high, z_s = D["state"], D["flags"], D["high"], D["z_s"]
LEARNING = learn_cfg is not None
ids = eps.index.to_numpy()
pos = {int(pid): i for i, pid in enumerate(ids)}


def pt(pid):
    return f"PT-{int(pid):05d}"


def sname(i, day):
    return STATES[state[i, day]]


def is_new(i, day):
    return day > 0 and state[i, day] > state[i, day - 1]


def next_key(i, day):
    s = sname(i, day)
    if s == "WATCH":
        return "WATCH_HIGH" if high[i] else "WATCH_LOW"
    return s


def pill(s, big=False):
    size = "font-size:1.05rem;padding:5px 16px;" if big else ""
    return f"<span class='ct-pill' style='background:{COLOR[s]};{size}'>{LABEL[s]}</span>"


def date_of(day):
    return DAY0 + dt.timedelta(days=int(day))


def change_text(i, day, m):
    v, n = D["today_val"][i, day, m], D["normal"][i, day, m]
    if np.isnan(n) or np.isnan(v):
        return "-"
    if LOG_SCALE[m]:
        return f"{100 * (v / n - 1):+.0f}%"
    return f"{v - n:+.1f} {UNIT[MEASURES[m]]}"


def notable(i, day, limit=None):
    """Measures that moved in the unhealthy direction, largest first."""
    zz = z_s[i, day]
    if np.all(np.isnan(zz)):
        return []
    cut = 0.5 if flags[i, day] else 1.0
    order = [m for m in np.argsort(-np.nan_to_num(zz, nan=-9)) if zz[m] > cut]
    order = order[:1] + [m for m in order[1:] if zz[m] > 1.0]      # extra measures only if clearly shifted
    return order[:limit] if limit else order


def cap(text):
    return text[:1].upper() + text[1:]


def what_changed(i, day):
    ms = notable(i, day, 2)
    return cap(", ".join(f"{SHORT[MEASURES[m]]} {change_text(i, day, m)}" for m in ms)) or "—"


def summary(i, day):
    female = eps["sex"].iloc[i] == "female"
    pron = "her" if female else "his"
    p = pred["p_ecg"].iloc[i]
    ecg = f"{'High' if high[i] else 'Low'}-risk clinic ECG ({100 * p:.0f}% probability of an abnormality)."
    a = int(D["run"][i, day])
    ch = [f"{SHORT[MEASURES[m]]} {change_text(i, day, m)}" for m in notable(i, day, 2)]
    if day < 2:
        wear = "Wearable data has just started; the twin is using population values for now."
    elif a > 0:
        wear = f"Readings have stayed away from {pron} usual range for {a} day{'s' if a > 1 else ''}: {', '.join(ch)}."
    elif ch:
        wear = f"Readings are mostly within {pron} usual range; watching {', '.join(ch)}."
    else:
        wear = f"Daily readings are within {pron} usual range."
    if LEARNING and day < LEARN_DAYS:
        wear += f" The twin is still learning {pron} normal ({100 * np.nanmean(D['learned'][i, day]):.0f}% known)."
    return f"{ecg} {wear}"


def events(i, day):
    out = []
    for t in range(1, day + 1):
        if state[i, t] != state[i, t - 1]:
            out.append((t, sname(i, t - 1), sname(i, t), what_changed(i, t)))
    return out


# ----------------------------------------------------------------------------------
# Sidebar: navigation and the demo controls
# ----------------------------------------------------------------------------------
if "view" not in st.session_state:
    st.session_state.view = "Worklist"
if "_goto" in st.session_state:
    st.session_state.view = st.session_state.pop("_goto")
    st.session_state.pop("worklist_table", None)
if "patient" not in st.session_state:
    st.session_state.patient = 160 if 160 in pos else int(ids[0])
st.session_state.setdefault("reviews", {})

st.sidebar.markdown("### 🫀 DilSe")
st.sidebar.radio("Go to", ["Worklist", "Patient", "How it works"], key="view", label_visibility="collapsed")
with st.sidebar.expander("Demo controls", expanded=True):
    day = st.slider("Replay day", 0, N_DAYS - 1, N_DAYS - 1,
                    help="Replays the 60 simulated days. A real clinic would only see today.")
    show_truth = st.toggle("Show simulation answers", value=False,
                           help="Reveals which patients had an injected (synthetic) episode and the true ECG label. "
                                "A real doctor would never see this.")
    st.caption("ECGs are real (PTB-XL test patients). Wearable data and episodes are simulated.")
st.sidebar.caption("**Research prototype. Not a medical device.** Suggestions support triage; "
                   "the treating clinician decides.")

today = date_of(day)
st.markdown(
    f"<div class='ct-bar'><div class='brand'>🫀 DilSe<span>Cardiac digital twin · remote monitoring</span></div>"
    f"<div class='when'>Cardiology follow-up clinic<br><b>{today:%A %d %b %Y}</b> · simulated day {day}</div></div>",
    unsafe_allow_html=True)


# ----------------------------------------------------------------------------------
# Screen 1: worklist
# ----------------------------------------------------------------------------------
def kpi(col, label, color, num, sub):
    col.markdown(f"<div class='ct-kpi'><div class='lab' style='color:{color}'>{label}</div>"
                 f"<div class='num'>{num:,}</div><div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


def worklist():
    st.header("Today's patients", anchor=False)
    now = state[:, day]
    new = np.array([is_new(i, day) for i in range(len(ids))])
    reviews = st.session_state.reviews
    reviewed = np.array([reviews.get(int(p), {}).get("day") == day for p in ids])
    c = st.columns(4)
    kpi(c[0], "Critical", COLOR["ESCALATE"], int((now == 2).sum()), f"{int((new & (now == 2)).sum())} new today")
    kpi(c[1], "Concern", COLOR["WATCH"], int((now == 1).sum()), f"{int((new & (now == 1)).sum())} new today")
    kpi(c[2], "Stable", COLOR["STABLE"], int((now == 0).sum()), "no action needed")
    need = now > 0
    kpi(c[3], "Reviewed today", "#12324a", int((reviewed & need).sum()), f"of {int(need.sum()):,} needing review")
    if LEARNING and day < LEARN_DAYS:
        st.caption(f"📈 Learning phase: the twin started from population values on day 0 and is learning each "
                   f"patient's own normal ({100 * np.nanmean(D['learned'][:, day]):.0f}% known on average today). "
                   f"From day {LEARN_DAYS} each patient's 28-day personal baseline takes over.")
    st.write("")

    f1, f2, f3, f4, f5 = st.columns([2.2, 1.2, 1.2, 1.1, 1.1])
    pick = f1.multiselect("Status", ["Critical", "Concern", "Stable"], default=["Critical", "Concern"])
    tier = f2.selectbox("Clinic ECG risk", ["All", "High", "Low"])
    search = f3.text_input("Find patient", placeholder="PT-00160")
    only_new = f4.toggle("New today only")
    hide_rev = f5.toggle("Hide reviewed")

    lo = max(0, day - 13)
    table = pd.DataFrame({
        "Status": [f"{DOT[STATES[s]]} {LABEL[STATES[s]]}" for s in now],
        "New": np.where(new, "● NEW", ""),
        "Patient": [pt(p) for p in ids],
        "Age / sex": [f"{int(a)} {'F' if s == 'female' else 'M'}" for a, s in zip(eps["age"], eps["sex"])],
        "Clinic ECG risk": (100 * pred["p_ecg"].to_numpy()).round(0).astype(int),
        "HR trend (14 d)": [list(np.round(D["vals"][i, lo:day + 1, 0], 1)) for i in range(len(ids))],
        "What changed": [what_changed(i, day) for i in range(len(ids))],
        "Alert days": D["run"][:, day],
        "Suggested next step": [NEXT[next_key(i, day)][2] for i in range(len(ids))],
        "Reviewed": np.where(reviewed, "✓", ""),
        "_s": now, "_new": new, "_score": np.nan_to_num(D["score"][:, day], nan=-9), "_rev": reviewed,
        "_high": high,
    })
    if show_truth:
        table["ANSWER: episode"] = np.where(eps["episode"] == 1, "+" + eps["rhr_rise_bpm"].astype(int).astype(str)
                                            + " bpm from day " + eps["start_day"].astype(str), "none")
    keep = table["Status"].str.split(" ", n=1).str[1].isin(pick)
    if tier != "All":
        keep &= table["_high"] == (tier == "High")
    if search.strip():
        keep &= table["Patient"].str.contains(search.strip().upper().replace("PT-", ""), regex=False)
    if only_new:
        keep &= table["_new"]
    if hide_rev:
        keep &= ~table["_rev"]
    view = (table[keep].sort_values(["_s", "_new", "Alert days", "_score"], ascending=[False, False, False, False])
            .drop(columns=["_s", "_new", "_score", "_rev", "_high"]))

    st.caption(f"{len(view):,} patients · most urgent first · **click a row** to open the patient")
    event = st.dataframe(
        view, hide_index=True, use_container_width=True, height=560,
        on_select="rerun", selection_mode="single-row", key="worklist_table",
        column_config={
            "Clinic ECG risk": st.column_config.ProgressColumn("Clinic ECG risk", min_value=0, max_value=100,
                                                               format="%d%%"),
            "HR trend (14 d)": st.column_config.LineChartColumn("HR trend (14 d)", width="small",
                                                                help="Resting heart rate over the last 14 days"),
            "Alert days": st.column_config.NumberColumn(help="Consecutive days the readings have stayed away from "
                                                             "the patient's usual range"),
            "New": st.column_config.TextColumn(width="small", help="Status is more urgent than yesterday"),
        })
    if event.selection.rows:
        st.session_state.patient = int(view.iloc[event.selection.rows[0]]["Patient"][3:])
        st.session_state._goto = "Patient"
        st.rerun()


# ----------------------------------------------------------------------------------
# Screen 2: patient
# ----------------------------------------------------------------------------------
def measure_chart(i, m_idx, ep_start):
    m = MEASURES[m_idx]
    days = np.arange(N_DAYS)
    past = days <= day
    df = pd.DataFrame({"day": days,
                       "value": np.where(past, D["vals"][i, :, m_idx], np.nan),
                       "usual": np.where(past, D["normal"][i, :, m_idx], np.nan),
                       "lo": np.where(past, D["lo"][i, :, m_idx], np.nan),
                       "hi": np.where(past, D["hi"][i, :, m_idx], np.nan),
                       "flag": past & flags[i]})
    x = alt.X("day:Q", scale=alt.Scale(domain=[0, N_DAYS - 1]), title=None)
    base = alt.Chart(df).encode(x=x)
    layers = [
        alt.Chart(df[df["flag"]]).mark_rect(color=COLOR["ESCALATE"], opacity=0.10)
        .encode(x="day:Q", x2=alt.X2("day_end:Q")).transform_calculate(day_end="datum.day + 1"),
        base.mark_area(color="#5b8db8", opacity=0.16).encode(y="lo:Q", y2="hi:Q"),
        base.mark_line(color="#5b8db8", strokeDash=[4, 3], strokeWidth=1.2).encode(y="usual:Q"),
        base.mark_line(color="#12324a", strokeWidth=1.6, point=alt.OverlayMarkDef(size=12, color="#12324a")).encode(
            y=alt.Y("value:Q", scale=alt.Scale(zero=False), title=UNIT[m]),
            tooltip=[alt.Tooltip("day:Q"), alt.Tooltip("value:Q", format=",.1f", title="value"),
                     alt.Tooltip("usual:Q", format=",.1f", title="usual")]),
    ]
    if LEARNING and day >= LEARN_DAYS:
        layers.append(alt.Chart(pd.DataFrame({"d": [LEARN_DAYS - 0.5]}))
                      .mark_rule(color="#9aa5ae", strokeDash=[2, 3]).encode(x="d:Q"))
    if ep_start is not None:
        layers.append(alt.Chart(pd.DataFrame({"d": [ep_start]}))
                      .mark_rule(color="purple", strokeDash=[2, 2]).encode(x="d:Q"))
    return alt.layer(*layers).properties(height=165, title=NICE[m]).configure_view(strokeWidth=0) \
        .configure_axis(gridColor="#eef1f3", labelColor="#5b6773", titleColor="#5b6773")


def ecg_figure(pid, show_saliency):
    import matplotlib.pyplot as plt
    src = ecg_source()
    if src is None or not src["has"](pid):
        return None
    sig, sal = src["get"](pid)
    t = np.arange(sig.shape[1]) / 250.0
    fig, axes = plt.subplots(12, 1, figsize=(14, 11), sharex=True)
    for k, ax in enumerate(axes):
        lo, hi = float(sig[k].min()), float(sig[k].max())
        pad = 0.1 * (hi - lo + 1e-6)
        if show_saliency and sal is not None:
            ax.imshow(sal[k][None, :], aspect="auto", cmap="Reds", vmin=0, vmax=1, alpha=0.55,
                      extent=[0, 10, lo - pad, hi + pad], interpolation="bilinear")
        ax.plot(t, sig[k], color="#222", linewidth=0.7)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_yticks([])
        ax.set_ylabel(LEADS[k], rotation=0, ha="right", va="center", fontsize=10)
        ax.grid(axis="x", color="#f0c8c8", linewidth=0.5)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel("time (s)")
    axes[-1].set_xticks(np.arange(0, 10.01, 1))
    fig.subplots_adjust(hspace=0.05, left=0.04, right=0.995, top=0.99, bottom=0.05)
    return fig


def pdf_summary(pid, i, note):
    """A one-page patient summary (for the file or a referral), drawn with matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = sname(i, day)
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.07, 0.95, "DilSe · patient summary", fontsize=18, weight="bold", color="#12324a")
    fig.text(0.07, 0.925, f"{pt(pid)} · {int(eps['age'].iloc[i])}-year-old {eps['sex'].iloc[i]} · "
                          f"{today:%d %b %Y} (simulated day {day})", fontsize=10.5, color="#444")
    fig.text(0.07, 0.885, f"Status: {LABEL[s]}", fontsize=14, weight="bold", color=COLOR[s])
    y = 0.855
    for line in textwrap.wrap(summary(i, day), 95):
        fig.text(0.07, y, line, fontsize=10.5)
        y -= 0.02
    nk = NEXT[next_key(i, day)]
    for line in textwrap.wrap(f"Suggested next step: {nk[0]}. {nk[1]}", 95):
        y -= 0.02
        fig.text(0.07, y, line, fontsize=10.5, weight="bold")
    y -= 0.045
    fig.text(0.07, y, "Measure", fontsize=10, weight="bold")
    fig.text(0.40, y, "Usual", fontsize=10, weight="bold")
    fig.text(0.58, y, "Last 3 days", fontsize=10, weight="bold")
    fig.text(0.78, y, "Change", fontsize=10, weight="bold")
    for m_idx, m in enumerate(MEASURES):
        y -= 0.022
        fmt = "{:,.0f}" if m == "steps" else "{:.1f}"
        fig.text(0.07, y, NICE[m], fontsize=10)
        fig.text(0.40, y, f"{fmt.format(D['normal'][i, day, m_idx])} {UNIT[m]}", fontsize=10)
        fig.text(0.58, y, f"{fmt.format(D['today_val'][i, day, m_idx])} {UNIT[m]}", fontsize=10)
        fig.text(0.78, y, change_text(i, day, m_idx), fontsize=10)
    ax = fig.add_axes([0.09, y - 0.30, 0.84, 0.24])
    d = np.arange(day + 1)
    ax.fill_between(d, D["lo"][i, :day + 1, 0], D["hi"][i, :day + 1, 0], color="#5b8db8", alpha=0.18, linewidth=0,
                    label="usual range")
    ax.plot(d, D["vals"][i, :day + 1, 0], color="#12324a", marker="o", markersize=2.5, linewidth=1.2,
            label="resting HR")
    for t in np.where(flags[i, :day + 1])[0]:
        ax.axvspan(t, t + 1, color=COLOR["ESCALATE"], alpha=0.1, linewidth=0)
    ax.set_xlim(0, N_DAYS - 1)
    ax.set_title("Resting heart rate (bpm), day 0 to today", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    y2 = y - 0.36
    fig.text(0.07, y2, f"Clinic ECG: {100 * pred['p_ecg'].iloc[i]:.0f}% probability of an abnormality "
                       f"({'high' if high[i] else 'low'} risk; 4-model ensemble, PTB-XL).", fontsize=10)
    if note:
        y2 -= 0.03
        for line in textwrap.wrap(f"Clinician note: {note}", 95):
            fig.text(0.07, y2, line, fontsize=10, style="italic")
            y2 -= 0.02
    fig.text(0.07, 0.04, "Research prototype, not a medical device. Wearable data are simulated. Suggestions support "
                         "triage only; the treating clinician decides.", fontsize=8, color="#666")
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    return buf.getvalue()


def patient():
    options = [pt(p) for p in ids]
    top_l, top_r = st.columns([1, 3])
    if top_l.button("← Back to worklist"):
        st.session_state._goto = "Worklist"
        st.rerun()
    choice = top_r.selectbox("Patient", options, index=options.index(pt(st.session_state.patient)),
                             label_visibility="collapsed")
    pid = int(choice[3:])
    st.session_state.patient = pid
    i = pos[pid]
    s = sname(i, day)
    female = eps["sex"].iloc[i] == "female"
    new_tag = "<span class='ct-new'>NEW TODAY</span>" if is_new(i, day) else ""
    st.markdown(f"<div style='font-size:2rem;font-weight:700;margin-bottom:0'>{pt(pid)} &nbsp;{pill(s, True)}{new_tag}</div>"
                f"<div class='ct-muted'>{int(eps['age'].iloc[i])}-year-old {'woman' if female else 'man'} · "
                f"{today:%a %d %b} · day {day}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='ct-summary'>{summary(i, day)}</div>", unsafe_allow_html=True)
    nk = NEXT[next_key(i, day)]
    st.markdown(f"<div class='ct-next' style='background:{TINT[s]};border-left:6px solid {COLOR[s]}'>"
                f"<div class='t'>Suggested next step: {nk[0]}</div><div class='d'>{nk[1]} "
                f"Decision support only; the treating clinician decides.</div></div>", unsafe_allow_html=True)

    # ----- actions: review, note, PDF
    reviews = st.session_state.reviews
    rev = reviews.get(pid, {})
    a1, a2, a3 = st.columns([1.1, 2.6, 1.1])
    note = a2.text_input("Note", value=rev.get("note", ""), placeholder="e.g. Phoned patient, feels well; "
                         "recheck Friday", label_visibility="collapsed", key=f"note_{pid}")
    if rev.get("day") == day:
        a1.success("Reviewed today ✓")
        if note != rev.get("note", ""):
            reviews[pid]["note"] = note
    elif a1.button("Mark as reviewed", type="primary", use_container_width=True):
        reviews[pid] = {"day": day, "note": note}
        st.rerun()
    a3.download_button("Download summary (PDF)", data=pdf_summary(pid, i, note),
                       file_name=f"{pt(pid)}_day{day}_summary.pdf", mime="application/pdf",
                       use_container_width=True)

    # ----- the twin's profile of this patient
    st.subheader("The twin's profile of this patient", anchor=False)
    c1, c2, c3 = st.columns([1, 1, 1.4])
    with c1.container(border=True):
        st.markdown("**Clinic ECG** · real, PTB-XL")
        folds = pred.iloc[i][["p_foldA", "p_foldB", "p_foldC", "p_foldD"]].astype(float)
        st.markdown(f"<div style='font-size:2rem;font-weight:700'>{100 * pred['p_ecg'].iloc[i]:.0f}%</div>"
                    f"<div class='ct-muted'>probability of an abnormal ECG</div>", unsafe_allow_html=True)
        st.markdown(f"Risk: **{'High' if high[i] else 'Low'}** (cut-off {100 * cfg['risk_threshold_p_ecg']:.0f}%)  \n"
                    f"The 4 models agree within {100 * (folds.max() - folds.min()):.0f} points"
                    + (" — treat as uncertain." if folds.max() - folds.min() > 0.3 else "."))
        if not high[i]:
            st.caption("A low-risk result does not rule out an abnormal ECG (the model finds about 4 in 5).")
        if show_truth:
            st.caption(f"ANSWER (cardiologist label): {'abnormal' if pred['label'].iloc[i] == 1 else 'normal'}")
    with c2.container(border=True):
        st.markdown("**What the twin knows**")
        pct = float(100 * np.nanmean(D["learned"][i, day]))
        if LEARNING and day < LEARN_DAYS:
            st.progress(min(max(pct / 100, 0.0), 1.0), text=f"{pct:.0f}% learned from {'her' if female else 'his'} "
                                                            f"own data")
            st.caption("Day 0 starts from population values for people like this patient; each day of their own "
                       f"data narrows the expected range. From day {LEARN_DAYS} a 28-day personal baseline is used.")
        else:
            st.progress(1.0, text="Personal baseline: 28 days of own data")
        tcfg = (learn_cfg if LEARNING and day < LEARN_DAYS else cfg)["high_risk" if high[i] else "low_risk"]
        st.markdown(f"Alert setting: **{'more sensitive' if high[i] else 'standard'}** "
                    f"({'high' if high[i] else 'low'}-risk ECG): an alert after **{tcfg['persistence']} days** in a "
                    f"row away from usual.")
        if show_truth:
            e = eps.iloc[i]
            st.caption("ANSWER: " + (f"synthetic episode of +{e['rhr_rise_bpm']:.0f} bpm from day {int(e['start_day'])}"
                                     if e["episode"] == 1 else "no episode injected"))
    with c3.container(border=True):
        st.markdown("**Today against usual** (last 3 days)")
        cols = st.columns(2)
        for k, m in enumerate(MEASURES):
            fmt = "{:,.0f}" if m == "steps" else "{:.1f}"
            v, n = D["today_val"][i, day, k], D["normal"][i, day, k]
            cols[k % 2].metric(NICE[m], f"{fmt.format(v)} {'/day' if m == 'steps' else UNIT[m]}",
                               None if np.isnan(n) else f"{change_text(i, day, k)} vs usual",
                               delta_color="inverse" if m == "resting_hr" else "normal")

    # ----- trends
    st.subheader("Trends up to today", anchor=False)
    st.caption("Dark line = daily value · blue band = usual range"
               + (f" (days 0–{LEARN_DAYS - 1}: the twin's estimate, wide at first and narrowing as it learns; "
                  f"then a 28-day personal baseline)" if LEARNING else "")
               + " · red shading = alert days" + (" · purple dashed = synthetic episode start" if show_truth else ""))
    ep_start = int(eps["start_day"].iloc[i]) if (show_truth and eps["episode"].iloc[i] == 1) else None
    g1, g2 = st.columns(2)
    for k, col in enumerate([g1, g2, g1, g2]):
        col.altair_chart(measure_chart(i, k, ep_start), use_container_width=True)

    # ----- status history and events
    st.subheader("Status history", anchor=False)
    cells = "".join(
        f"<div title='day {d}: {LABEL[sname(i, d)]}' style='flex:1;height:22px;"
        f"background:{COLOR[sname(i, d)] if d <= day else '#eef1f3'};border-right:1px solid white'></div>"
        for d in range(N_DAYS))
    ticks = "".join(f"<div style='flex:1;font-size:11px;color:#6a7682'>{d if d % 10 == 0 else ''}</div>"
                    for d in range(N_DAYS))
    st.markdown(f"<div style='display:flex;border-radius:6px;overflow:hidden'>{cells}</div>"
                f"<div style='display:flex'>{ticks}</div>", unsafe_allow_html=True)
    ev = events(i, day)
    if ev:
        st.markdown("".join(
            f"<div class='ct-event'><b>Day {t}</b> ({date_of(t):%d %b}) · {LABEL[a]} → "
            f"<span style='color:{COLOR[b]};font-weight:700'>{LABEL[b]}</span>"
            + (f" · {w}" if w != "—" else "") + "</div>" for t, a, b, w in reversed(ev)), unsafe_allow_html=True)
    else:
        st.caption(f"No status change so far: {LABEL[s]} since day 0.")

    # ----- ECG
    with st.expander("12-lead clinic ECG (10 s, real PTB-XL recording)"):
        src = ecg_source()
        if src is None or not src["has"](pid):
            st.info("This patient's ECG trace is not included in the hosted demo. Run locally with "
                    "data/processed/X_test.npy to see every patient.")
        else:
            shade = st.toggle("Shade where the model's score is most sensitive (saliency)", value=True,
                              help="Not a diagnosis and not the model's reasoning: it shows where small changes in "
                                   "the trace would move the risk score most (saliency.py).")
            fig = ecg_figure(pid, shade)
            st.pyplot(fig, use_container_width=True)
            import matplotlib.pyplot as plt
            plt.close(fig)

    # ----- technical details for reviewers
    with st.expander("Technical details"):
        rows = []
        for k, m in enumerate(MEASURES):
            zz = z_s[i, day, k]
            rows.append({"Measure": NICE[m], "Shift (SD, unhealthy +)": None if np.isnan(zz) else round(float(zz), 2),
                         "Unhealthy direction": f"{BAD_WAY[m]} usual"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        sc = D["score"][i, day]
        phase = "learning twin, D-24" if LEARNING and day < LEARN_DAYS else "locked monitor, D-17/D-18"
        tcfg = (learn_cfg if LEARNING and day < LEARN_DAYS else cfg)["high_risk" if high[i] else "low_risk"]
        st.markdown(f"Combined wearable score today: **{'-' if np.isnan(sc) else f'{sc:+.2f}'} SD** ({phase}). "
                    f"Alert rule: score above **{tcfg['threshold']}** on **{tcfg['persistence']}** days in a row. "
                    f"ECG ensemble models: " + ", ".join(f"{100 * float(pred[c].iloc[i]):.0f}%"
                                                        for c in ["p_foldA", "p_foldB", "p_foldC", "p_foldD"]))


# ----------------------------------------------------------------------------------
# Screen 3: how it works
# ----------------------------------------------------------------------------------
def how_it_works():
    st.header("How DilSe decides", anchor=False)
    st.markdown(
        "DilSe combines two streams into one status per patient per day.\n\n"
        "1. **Clinic ECG (real data).** A neural network (4 models averaged) reads the 12-lead ECG and gives the "
        "probability of an abnormality. At or above the cut-off the patient is **high risk**.\n"
        "2. **Daily wearable readings (simulated).** Resting heart rate, heart-rate variability, sleep and steps are "
        "compared with **the patient's own usual range**. On day 0 the twin starts from population values for people "
        "like this patient and learns their normal day by day; from day 35 it uses a 28-day personal baseline. An "
        "**alert** needs the readings to stay away from usual for several days in a row.\n"
        "3. **The twin.** The ECG risk sets how sensitive the alert is (high-risk patients are alerted sooner), and "
        "risk plus alert gives the status:")
    c = st.columns(3)
    for col, s in zip(c, ["STABLE", "WATCH", "ESCALATE"]):
        col.markdown(f"<div class='ct-next' style='background:{TINT[s]};border-left:6px solid {COLOR[s]}'>"
                     f"<div class='t'>{LABEL[s]}</div><div class='d'>{MEANING[s]}<br><b>Next step:</b> "
                     f"{NEXT['WATCH_HIGH' if s == 'WATCH' else s][0]}</div></div>", unsafe_allow_html=True)

    st.subheader("How well it works", anchor=False)
    res = json.load(open(FINAL / "final_results.json"))["ecg"]
    m, lt = res["ECG only (FINAL MODEL)"], res["locked_threshold"]
    k = st.columns(4)
    k[0].metric("ECG model AUROC", f"{m['auroc']:.3f}", help="2,145 unseen test patients, real ECGs")
    k[1].metric("Abnormal ECGs found", f"{100 * lt['sensitivity']:.0f}%", help="Sensitivity at the locked cut-off")
    k[2].metric("Normal ECGs cleared", f"{100 * lt['specificity']:.0f}%", help="Specificity at the locked cut-off")
    k[3].metric("False alerts budget", "0.5 / month", help="Per patient per month: the budget the alerts were tuned "
                                                           "to (synthetic wearable data)")
    st.caption("ECG results: real PTB-XL test patients, scored once. Wearable results: simulated data; on real "
               "people (PMData) the alert rate was 0.63 per person-month.")

    st.subheader("Limits", anchor=False)
    st.markdown(
        "- The ECG model detects a **current** abnormality; it does not predict a future event.\n"
        "- The ECGs are from a German hospital (1989–1996); performance in Indian patients is unknown.\n"
        "- It misses about 1 in 5 abnormal ECGs, so a low-risk result does not rule anything out.\n"
        "- **All wearable data and deterioration episodes are simulated.**\n"
        "- Research prototype; not a medical device; no diagnosis or treatment advice.")
    with st.expander("Locked settings"):
        st.markdown(
            f"ECG risk cut-off {cfg['risk_threshold_p_ecg']:.3f} (90% specificity on validation). "
            f"Locked monitor (days 35–59): low risk threshold {cfg['low_risk']['threshold']}, "
            f"k = {cfg['low_risk']['persistence']}; high risk {cfg['high_risk']['threshold']}, "
            f"k = {cfg['high_risk']['persistence']}."
            + (f" Learning twin (days 0–34, D-24): low risk {learn_cfg['low_risk']['threshold']}, "
               f"k = {learn_cfg['low_risk']['persistence']}; high risk {learn_cfg['high_risk']['threshold']}, "
               f"k = {learn_cfg['high_risk']['persistence']}." if LEARNING else "")
            + " All settings were locked on validation patients; details in DECISIONS.md.")


if st.session_state.view == "Worklist":
    worklist()
elif st.session_state.view == "Patient":
    patient()
else:
    how_it_works()
