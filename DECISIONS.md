# Decision Log — Cardiovascular Risk Digital Twin

**Challenge:** Digital Twin Challenge 2026 (Happiest Health, on Unstop)
**Condition:** Cardiovascular risk
**Working deadline:** 14 Oct 2026 (hard deadline treated as 17 Oct 2026; page states 20 Oct 2026)
**Log started:** 03 Oct 2026

This file records every major design decision on the project: what we were deciding, the options we weighed, what we chose, and what we gave up. Entries are never deleted. If a decision changes, the old entry is marked **Changed** or **Reversed** and links to the new one.

---

## Index

| ID | Title | Date | Decided by | Status |
|----|-------|------|------------|--------|
| D-1 | Prediction target: binary normal vs abnormal ECG (PTB-XL) | 03 Oct 2026 | Me, on mentor recommendation | Active |
| D-2 | Dataset PTB-XL v1.0.3 at 500 Hz, processed one record at a time | 03 Oct 2026 | Me, on mentor recommendation | Changed (sampling rate, see D-9) |
| D-3 | Label rules: inclusive, any abnormal code = 1, all likelihoods kept | 04 Oct 2026 | Me, on mentor recommendation | Active |
| D-4 | Study population: adults 18+, age 300 recoded to 90 | 04 Oct 2026 | Me, on mentor recommendation | Active |
| D-5 | Stay with PTB-XL; CODE-15% considered and rejected | 04 Oct 2026 | Me, on mentor recommendation | Active |
| D-6 | ECG approach: staged hybrid (CNN on raw ECG first, clinical features added only if they help) | 04 Oct 2026 | Me, on mentor recommendation | Active |
| D-7 | CNN architecture: small 1D ResNet, built up from a plain CNN | 04 Oct 2026 | Me, on mentor recommendation | Changed (see D-10) |
| D-8 | Repository layout: keep scripts in place, add docs/, requirements, .gitignore (A) | 06 Oct 2026 | Me, on mentor recommendation | Active |
| D-9 | CNN input: 100 Hz (downsampled from 500 Hz), chosen by a pre-agreed rule | 04 Oct 2026 | Me, based on my own experiment | Active |
| D-10 | ECG model: keep the plain CNN; ResNet tested and did not clear the pre-agreed bar | 04 Oct 2026 | Me, applying a rule set in advance | Active |
| D-11 | Fusion: cross-fitted stacking (4 out-of-fold CNNs + logistic regression on ECG score, age, sex) | 04 Oct 2026 | Me, on mentor recommendation | Active |
| D-12 | Final risk score: ECG-only 4-CNN ensemble (fused model lost the pre-agreed calibration check) | 04 Oct 2026 | Me, applying a rule set in advance | Active |
| D-13 | Wearable stream: hybrid simulation (generator parameters estimated from an open dataset) | 04 Oct 2026 | Me, on mentor recommendation | Active |
| D-14 | Wearable recipe sources: PMData + MMASH + cited population ranges | 05 Oct 2026 | Mentor, with my approval (delegated) | Active (licence question resolved by D-14a) |
| D-14a | Keep PMData despite its CC BY-NC 4.0 licence; declare it openly | 05 Oct 2026 | Me | Active |
| D-15 | Personal baselines from age/sex population studies (B); ECG-anchored resting HR (C) as stretch goal | 05 Oct 2026 | Me, on mentor recommendation | Active |
| D-15a | HRV day-to-day variability from a published study (Hannon et al. 2025); memory assumed | 05 Oct 2026 | Me, on mentor recommendation | Active |
| D-16 | Deterioration episodes: a grid of sizes reported as detection limits | 05 Oct 2026 | Me, on mentor recommendation | Active |
| D-17 | Wearable monitor: rolling personal baseline + persistence (B), fixed baseline (A) as comparison | 05 Oct 2026 | Me, on mentor recommendation | Active |
| D-18 | Twin state: ECG risk tier sets monitor sensitivity, then a Stable/Watch/Escalate table | 05 Oct 2026 | Me, on mentor recommendation | Active |
| D-19 | Dashboard tool: Streamlit | 05 Oct 2026 | Me, on mentor recommendation | Active |
| D-20 | Dashboard content: worklist + patient detail with a 'why' panel | 05 Oct 2026 | Me, on mentor recommendation | Active |
| D-21 | Improvement package after competitor review: real-data monitor check, simple-rule baseline, integrity tests, ECG saliency | 06 Oct 2026 | Me, on mentor recommendation | Active |
| D-22 | Doctor-first dashboard redesign (Streamlit) | 06 Oct 2026 | Me, on mentor recommendation | Active |
| D-23 | Code licence: MIT | 06 Oct 2026 | Me, on mentor recommendation | Active |
| D-24 | Learning twin: monitoring from day 0, shrinking from a population prior to the personal normal | 14 Oct 2026 | Me | Active |
| D-25 | Product-style app: clinical language, workflow features, twin profile, polish, free hosting | 14 Oct 2026 | Me | Active |
| D-26 | ECG heart age: a second CNN estimates how old the heart looks (R1 pass, R2 fail: not shown in the app) | 15 Oct 2026 | Me | Closed (negative result) |

---

## Project context (confirmed)

- Team: **solo** (confirmed 03 Oct 2026)
- Skills: Python comfortable, classical ML comfortable, PyTorch never used (confirmed 03 Oct 2026)
- Laptop: 16 GB RAM, no GPU needed (Colab as fallback) (confirmed 03 Oct 2026)

## Progress log

- **04 Oct 2026, data downloaded and verified.** PTB-XL v1.0.3 was downloaded through the AWS open-data mirror listed on the PhysioNet page (`s3://physionet-open/ptb-xl/1.0.3/`), skipping the 100 Hz folder. Local checks: 21,799 `.dat` and 21,799 `.hea` files in `records500`, matching the record count on the PhysioNet page (Abstract and Data Description). `ptbxl_database.csv` is 6,594,879 bytes, consistent with the 6.3 MB listed in the page's "Files" table. `scp_statements.csv`, `LICENSE.txt` and `SHA256SUMS.txt` are present.

- **04 Oct 2026, first model trained: plain CNN, ECG only, 500 Hz** (train_cnn.py, seed 42, 189,889 parameters). Best validation epoch 19 of 20: **val AUROC 0.938 · AUPRC 0.957 · sensitivity at 90% specificity 0.825**. Not converged: it reached the 20-epoch cap without early stopping. CPU time about 4–6.5 minutes per epoch, roughly 1.5 hours in total. Note: AUPRC must be read against the 57.6% abnormal share, which is the AUPRC of a random model. Validation scores are slightly optimistic because the best epoch was chosen on validation; the test set is untouched.

- **05 Oct 2026, plan checked against the official Unstop guidelines** (text supplied by me from the challenge page):
  - Data rule ("Navigating data constraints"): teams must use "anonymized, open-source, or synthetic datasets for training". For EHRs, anonymized clinical databases are explicitly allowed (MIMIC-IV given as an example); for sensor data, teams may use open-source wearable datasets or generate their own synthetic time series. **PTB-XL (anonymized, open) and a simulated wearable stream both comply.**
  - No requirement that the data be Indian; only the *condition* must be prevalent in India (cardiovascular risk is listed as an example).
  - Team size 1–4 full-time students; solo entry is allowed.
  - **Requirement to watch:** "a working algorithmic model that ingests these two streams and successfully predicts an adverse event". The final risk score (D-12) uses the ECG only, so the combination of the ECG risk score with the wearable stream must itself be an algorithm, not just two panels side by side on the dashboard.
  - **Gap to watch:** the static-data examples listed are "demographics, past diagnoses, lab results, genetic markers". Our real static data is demographics plus the clinic ECG (L-2, L-4).
  - Minor inconsistency in the guidelines: Phase 1 says the jury finalises the top 5, while Phase 2 says the top 10 present virtually.

## Open questions (not yet decided)

- D-8, repository structure and naming: deferred until the model was finished; **decided 06 Oct 2026 (see D-8 entry below)**. Work continues locally in `dt-cardio` and everything is pushed to GitHub at the end. I chose no local git for now, accepting the risk of losing work between sessions.

- How to cover "labs / diagnoses" EHR fields without leaking simulated data into the classifier

---

## Running limitations

*(Updated as decisions are made. This section feeds the README "Limitations" section.)*

- **L-1 (from D-1):** The model detects a *current* cardiac abnormality on the ECG. It does not predict a *future* adverse event, and no open dataset available to us links these patients to later outcomes.
- **L-2 (from D-1):** Real static data for the same patients is limited to age, sex, height and weight. There are no labs or diagnosis history.
- **L-4 (from data exploration):** Height is missing for 68.0% of records and weight for 56.8%, so BMI is unavailable for most patients. In practice the real static features are age and sex.
- **L-5 (from data exploration):** 57.6% of the labelled records are abnormal. This is a hospital ECG population, far sicker than the general public, so the predicted probabilities and alert rates will not transfer directly to population screening.
- **L-6 (from D-11):** Adding age and sex to the ECG score gave no measurable improvement in discrimination on validation (ΔAUROC −0.0007, 95% CI −0.0025 to +0.0010). The static data mainly contributes explanation and calibration, not accuracy.
- **L-7 (from D-12):** The final risk score is mildly underconfident on validation (actual abnormal rate exceeds predicted risk by up to 0.08 in mid-range bins; ECE 0.029). Displayed probabilities should be read as slightly conservative. Confirmed on test: mean predicted risk 0.538 vs actual abnormal share 0.576; ECE 0.038; Brier 0.101.
- **L-8 (from final test):** At the locked threshold the model misses 242 of 1,236 abnormal test ECGs (sensitivity 0.804). A STABLE or low-risk label therefore does not rule out an abnormal ECG; the dashboard must say so.
- **L-10 (from D-21):** On real people (PMData, no labelled episodes, HRV unavailable) the locked monitor raised 0.63 alarms per person-month, about 1.3× the 0.5 budget it was tuned to on synthetic data. Real-world false alarms are likely to be higher than the synthetic results suggest.
- **L-11 (from D-21):** In the synthetic test, a resting-HR-only monitor detects almost as much as the 4-measure monitor (mean 89.5% vs 90.0% at the same false-alarm budget). The other three measures add little *in our simulation*, partly because the simulated episodes change them only modestly (D-16).
- **L-9 (from D-13 to D-18):** All wearable and twin-state results are measured on simulated data and simulated episodes. They show that the pipeline works and how alerts are distributed; they are not evidence of clinical detection performance.
- **L-12 (from D-24):** The learning twin's synthetic results (days 0–34) are optimistic, because its population prior comes from the same recipe that generated the synthetic patients. Its real-world check (PMData) covers false alarms only, not detection.
- **L-3 (from D-1):** PTB-XL was recorded in Germany between 1989 and 1996. The population is not Indian, so performance on Indian patients is unknown.

---

## Entry template

Copy this block for each new decision.

```
### D-<number>: <short title>
- Date:
- Context: what question we were answering
- Options considered:
  - Option A: <description>
    - Pros:
    - Cons:
    - Time cost:
  - Option B: <description>
    - Pros:
    - Cons:
    - Time cost:
  - Option C (if any): <description>
    - Pros:
    - Cons:
    - Time cost:
- Decision: what we chose, and who decided (me, or mentor with my approval)
- Reasoning: why, including trade-offs and what we gave up
- Risks and limitations: what could go wrong, or what a judge might question
- Status: Active / Changed (see D-x) / Reversed (see D-x)
```

---

## Decisions

### D-1: Prediction target — binary normal vs abnormal ECG
- **Date:** 03 Oct 2026
- **Context:** The challenge asks the model to "predict an adverse cardiovascular event or flag a high-risk anomaly". We needed one clear prediction target with real labels. No open dataset has the same patients with EHR data, ECG, wearables *and* future outcomes, so every option involves a trade-off.
- **Options considered:**
  - **Option A: PTB-XL binary, normal (NORM only) vs any abnormality (MI, STTC, CD, HYP)**
    - Pros: real cardiologist labels. The static data (age, sex, height, weight) and the ECG belong to the *same* patients, so fusion is genuine. One target and simple metrics. Easiest to explain.
    - Cons: detects a current abnormality, not a future event. Thin static features. Some label-cleaning rules are needed.
    - Time cost: about 1 day for data and labels.
  - **Option B: PTB-XL five diagnostic superclasses (multi-label)**
    - Pros: more clinically informative, since it tells the doctor which kind of abnormality.
    - Cons: metrics multiply (per-class plus macro/micro averages). The rarest class (HYP) will have weaker performance. Five explanations per patient. Much harder to explain in the video.
    - Time cost: about 2–3 days more than A.
  - **Option C: Framingham-style 10-year coronary heart disease risk**
    - Pros: a true future-event prediction, with rich risk-factor features (BP, cholesterol, smoking, diabetes).
    - Cons: those patients have no ECGs, so fusion would mean pairing them with other people's ECGs. Any fusion result would be meaningless or an artefact. It is a US cohort, so calibration for Indian patients is questionable. Licence and provenance of the common Kaggle copy are unverified.
    - Time cost: about 1 day for the model, but the fusion story is indefensible.
- **Decision:** Option A. Decided by me, on mentor recommendation. B may be added later as a *dashboard-only* breakdown that doesn't change the headline metrics.
- **Reasoning:** A is the only option where static and ECG data come from the same real patients, so the fusion gain (or lack of one) is a real finding. Its main weakness, that it is not a future prediction, can be defended openly using the challenge's own "flag a high-risk anomaly" wording. We gave up a true future-event target and rich static features in exchange for honest fusion and a scope one person can finish.
- **Risks and limitations:** A judge may ask "where is the prediction?". Answer: we flag patients with an underlying abnormality, which is a recognised risk marker, and the wearable monitor watches them for deterioration. The static-only baseline will be weak. See L-1, L-2 and L-3.
- **Status:** Active

### D-2: Dataset and sampling rate — PTB-XL v1.0.3 at 500 Hz
- **Date:** 03 Oct 2026
- **Context:** D-1 fixed PTB-XL as the source of the static data, the ECG and the labels. The download includes both 100 Hz and 500 Hz signals, so we needed to choose which one to extract ECG features from.
- **Facts verified on the official page** (physionet.org/content/ptb-xl/1.0.3, checked 03 Oct 2026):
  - Licence: CC BY 4.0 ("Access" section).
  - Access: Open Access, no credentialing needed ("Access" section).
  - Size: 21,799 recordings from 18,869 patients (Abstract and Data Description). The page's meta description says 21,801; we will count rows after download.
  - Recommended split: folds 1–8 train, 9 validation, 10 test, with patients kept within one fold (Data Description, "Cross-validation Folds").
  - Download: 1.7 GB ZIP, 3.0 GB uncompressed, containing both sampling rates ("Files" section).
  - Ages above 89 are stored as roughly 300 for privacy (Data Preprocessing), so they must be cleaned.
- **Options considered:**
  - **Option A: 100 Hz**
    - Pros: smaller and faster to process. Fine for heart rate and rhythm features.
    - Cons: one sample every 10 ms makes interval features (QRS, QT, PR) coarse, which weakens detection of conduction disturbances.
    - Time cost: fastest.
  - **Option B: 500 Hz, processed one record at a time**
    - Pros: one sample every 2 ms gives precise interval features. Processing per record keeps memory use low on 16 GB RAM.
    - Cons: feature extraction is roughly 5× slower (to be timed on a 100-record sample). It runs once, with features cached to disk.
    - Time cost: about half a day more, mostly machine time.
  - **Option C: develop on 100 Hz, run the final extraction on 500 Hz**
    - Pros: fast iteration plus precise final features.
    - Cons: two runs, and a risk of code behaving differently at the two rates.
- **Decision:** Option B. Decided by me, on mentor recommendation. *(Sampling rate later changed for CNN training; see D-9. The 500 Hz files remain the stored source.)*
- **Reasoning:** We are using hand-crafted ECG features, so the quality of those features *is* the quality of the ECG model. The extra cost is machine time, not working time. The decision is cheap to reverse because both rates are in the same download.
- **Risks and limitations:** Extraction may be slower than estimated; if so, we fall back to Option C. Required citations: the dataset DOI (10.13026/kfzx-aw45), Wagner et al. 2020 (Scientific Data), and the standard PhysioNet citation, all listed on the dataset page.
- **Status:** Active

### D-3: Label-cleaning rules — inclusive, all likelihoods kept
- **Date:** 04 Oct 2026
- **Context:** D-1 defined the target as normal vs any abnormality. We needed exact rules for turning PTB-XL's diagnostic codes into a 0/1 label. Two facts from the PhysioNet page (Data Description) shaped the options: one ECG can carry several codes, so a record can be marked NORM and abnormal together; and each code's likelihood "is set to 0 if unknown", so a likelihood threshold would wrongly discard real diagnoses whose confidence wasn't recorded.
- **Rules common to all options:** codes are mapped to superclasses using `scp_statements.csv` (diagnostic codes only). Label 0 = the record's only superclass is NORM. Records with no diagnostic superclass at all are dropped.
- **Options considered:**
  - **Option A: Strict, drop records marked both NORM and abnormal**
    - Pros: cleanest labels.
    - Cons: removes exactly the borderline cases a clinic would see, which makes the task easier and inflates the scores.
    - Time cost: none extra.
  - **Option B: Inclusive, any abnormal superclass (MI, STTC, CD, HYP) means label 1; all likelihoods kept**
    - Pros: matches screening logic. Keeps the hard cases, so the scores stay honest. Avoids the likelihood-0 trap.
    - Cons: slightly noisier labels and slightly lower scores.
    - Time cost: none extra.
  - **Option C: Inclusive, but ignore abnormal codes below a likelihood threshold (0 treated as unknown and kept)**
    - Pros: filters out tentative diagnoses.
    - Cons: the threshold is an arbitrary choice that must be defended, and it adds explanation time.
    - Time cost: about 2 hours.
- **Decision:** Option B, using all likelihoods. Decided by me, on mentor recommendation.
- **Reasoning:** "A patient is labelled abnormal if a cardiologist recorded any abnormal diagnostic statement" is simple, clinically sensible and easy to defend. We accept slightly lower scores in exchange for not hiding the hard cases.
- **Risks and limitations:** Some label noise from low-confidence or mixed annotations. The number of records affected by each rule (mixed NORM + abnormal, dropped for no diagnosis, likelihood 0) will be counted by `explore_data.py` and recorded here.
- **Counts from our data** (explore_data.py, run 04 Oct 2026):
  - 21,799 records from 18,869 patients loaded, matching the PhysioNet page.
  - Superclass counts match the PhysioNet table exactly (NORM 9,514 · MI 5,469 · STTC 5,235 · CD 4,898 · HYP 2,649), confirming the label mapping is correct.
  - Dropped (no diagnostic superclass): 411.
  - Mixed NORM + abnormal, labelled 1: 445. Check: 9,069 pure normal + 445 mixed = 9,514 NORM records ✓.
  - Records with a diagnostic code at likelihood 0, kept: 213. A likelihood threshold would have wrongly removed these.
  - Final labelled set: 21,388 records, of which 9,069 are normal (42.4%) and 12,319 abnormal (57.6%).
  - Splits: train 17,084 · validation 2,146 · test 2,158, with a similar class balance in each (about 42% normal).
  - Patients appearing in more than one fold: 0 (no patient-level leakage).
- **Status:** Active

### D-4: Study population — adults only, age 300 recoded to 90
- **Date:** 04 Oct 2026
- **Context:** Exploration showed ages from 2 years upwards, plus 293 records with age = 300 (the privacy code for "over 89", per the PhysioNet page, Data Preprocessing). Children's ECGs follow different normal ranges, and our use case is adult lifestyle-related cardiovascular risk in India.
- **Options considered:**
  - **Option A: Keep all ages**
    - Pros: maximum data.
    - Cons: mixes paediatric and adult ECG patterns. Clinically hard to defend for an adult cardiovascular risk tool.
    - Time cost: none.
  - **Option B: Adults only (age 18 or over); age 300 recoded to 90**
    - Pros: matches the use case and is one sentence to defend. Recoding to 90 keeps the over-89 patients without inventing ages.
    - Cons: loses the paediatric records (exact count to be recorded when the filter is applied). Over-89s are compressed to a single value.
    - Time cost: minimal.
  - **Option C: Keep all ages and add an is_child flag**
    - Pros: keeps all data.
    - Cons: extra complexity for a population the twin will not serve.
    - Time cost: small.
- **Decision:** Option B. Decided by me, on mentor recommendation.
- **Reasoning:** Clinical coherence matters more than a few hundred extra records for a tool aimed at adults.
- **Risks and limitations:** Records excluded as under 18 will be counted and reported here. Ages of 90 represent "90 or older".
- **Counts from our data** (prepare_data.py, run 04 Oct 2026):
  - Ages recoded from 300 to 90: 285. The other 8 of the 293 privacy-coded records had already been dropped for having no diagnostic superclass.
  - Dropped, under 18: 133, of which 35 abnormal. 21,388 − 133 = 21,255 records remain.
  - Final splits: train 16,973 (57.8% abnormal) · validation 2,137 (57.6%) · test 2,145 (57.6%). Patients in more than one split: 0.
  - Signal problems (wrong shape or NaN): 0.
  - Train-only per-lead statistics: means about 0 mV; standard deviations 0.14–0.33 mV, largest in V2–V3.
- **Status:** Active

### D-5: Dataset, final check — stay with PTB-XL; CODE-15% considered and rejected
- **Date:** 04 Oct 2026
- **Context:** Exploration showed that PTB-XL's real static data is effectively only age and sex (L-4). We checked whether switching to CODE-15%, a Brazilian ECG dataset with mortality follow-up, would be better.
- **Facts verified on the official Zenodo page** (zenodo.org/records/4916206, checked 04 Oct 2026):
  - Licence: CC BY 4.0 ("Rights").
  - 345,779 exams from 233,770 patients, collected 2010–2016 by the Telehealth Network of Minas Gerais ("Description").
  - Static fields: age and is_male only ("Description").
  - Labels: six abnormalities plus `normal_ecg`, the latter from an automatic annotation system ("Description").
  - Outcome: `death` and `timey` (follow-up time), filled in only on each patient's first exam ("Description").
  - Signals: 400 Hz, 7 or 10 s, zero-padded to 4,096 samples, HDF5 format ("Description").
  - Total download 46.3 GB across 18 ZIP parts ("Files"). Local free disk space was 15.5 GB.
- **Options considered:**
  - **Option A: Stay with PTB-XL**
    - Pros: cardiologist-reviewed labels, already downloaded and verified, D-1 to D-4 already done.
    - Cons: no future outcome, thin static data.
    - Time cost: none.
  - **Option B: Switch to CODE-15%**
    - Pros: a real future outcome (mortality), and a primary-care telehealth population.
    - Cons: static data is no richer (age and sex only). The outcome is all-cause death, not cardiovascular. Labels are automatic. The full dataset does not fit on disk. Variable follow-up requires survival analysis. D-1 to D-4 would need redoing.
    - Time cost: about 3–4 days.
  - **Option C: PTB-XL now, CODE-15% as a later stretch**
    - Pros: keeps the core safe and leaves the door open.
    - Cons: realistically there will be no time left for it.
    - Time cost: 1–2 days, if used.
- **Decision:** Option A. Decided by me, on mentor recommendation.
- **Reasoning:** CODE-15% does not fix the thin static data that triggered the check, and it adds new weaknesses: all-cause outcome, automatic labels, size, and survival-analysis complexity. We gave up a true future-outcome target.
- **Risks and limitations:** The jury may still ask about future prediction. Answer: CODE-15% was evaluated and rejected for the reasons above. The thin static data is addressed by framing the resting clinic ECG as part of the clinical record, alongside demographics.
- **Status:** Active

### D-6: ECG approach — staged hybrid, CNN first
- **Date:** 04 Oct 2026
- **Context:** The ECG carries almost all the predictive signal, so we needed to decide how the model reads it. I (the student) pointed out a flaw in a features-only approach: wave-detection tools fail most often on abnormal ECGs, which is exactly where the information is.
- **Options considered:**
  - **Option A: Clinical features (NeuroKit2) + XGBoost only**
    - Pros: easy to explain, and the explanations read like clinical reasoning.
    - Cons: detection failures concentrate in abnormal ECGs, so features are wrong or missing where it matters. XGBoost could learn "missing features → abnormal", a shortcut that also flags noisy normal ECGs.
    - Time cost: about 1.5 days.
  - **Option B: 1D-CNN on the raw ECG only**
    - Pros: no detection step, so it learns directly from abnormal waveforms. The standard approach on PTB-XL (see the benchmark paper by Strodthoff et al. 2021, reference 4 on the PhysioNet page).
    - Cons: explanations limited to saliency maps over the trace; no named measurements for the doctor. Requires learning PyTorch.
    - Time cost: about 3 days.
  - **Option C: Staged hybrid**: (1) CNN on the raw ECG as the main ECG model; (2) NeuroKit2 measurements shown on the dashboard; (3) those measurements tested as extra fusion inputs, kept only if they improve validation scores.
    - Pros: the model does not depend on wave detection, the doctor still sees familiar measurements, and every addition must be proven by ablation.
    - Cons: the most components. Needs discipline to finish step 1 before step 3.
    - Time cost: about 3 days for steps 1–2, plus about 1 day for step 3.
- **Decision:** Option C. Decided by me, on mentor recommendation.
- **Reasoning:** Removes the detection-failure weakness while keeping a clinically readable dashboard. The additions are evidence-driven rather than assumed.
- **Risks and limitations:** PyTorch learning curve within a 10-day window. Measurements shown on the dashboard are *measured values*, not the model's reasoning, and must be labelled that way. D-2 (500 Hz) was chosen for hand-crafted features and may be revisited for the CNN through a quick experiment. Licences for PyTorch and NeuroKit2 still to be verified by me on their official pages.
- **Status:** Active

### D-7: CNN architecture — small 1D ResNet, built up from a plain CNN
- **Date:** 04 Oct 2026
- **Context:** D-6 made a CNN on the raw 12-lead ECG the main ECG model. We needed to choose its architecture.
- **Options considered:**
  - **Option A: Plain CNN (3–4 convolution blocks + pooling)**
    - Pros: simplest to write and explain.
    - Cons: deeper plain CNNs become hard to train, which limits capacity.
    - Time cost: about 1 day.
  - **Option B: Small 1D ResNet (convolution blocks with skip connections)**
    - Pros: skip connections make deeper networks train reliably. Residual-style networks are well established for 12-lead ECGs (to be confirmed in Ribeiro et al. 2020 and the PTB-XL benchmark paper, Strodthoff et al. 2021, before being claimed in the slides).
    - Cons: about twice the code of A, and one more concept to explain.
    - Time cost: about 1.5 days.
  - **Option C: Pretrained ECG model, fine-tuned**
    - Pros: potentially the strongest scores.
    - Cons: licence and training data must be vetted. If it was trained on PTB-XL, it may have seen our test set (uncontrollable leakage). A black box we cannot fully explain.
    - Time cost: unpredictable.
- **Decision:** Option B, built in two steps: first a plain CNN as a learning exercise and sanity check, then skip connections added. Decided by me, on mentor recommendation.
- **Reasoning:** Learning PyTorch on the simple version first lowers risk. The before/after comparison gives us our own evidence for why skip connections help. Option C's leakage risk conflicts with honest evaluation.
- **Risks and limitations:** Training time on CPU is unknown until measured; Google Colab is the fallback. Hyperparameters (depth, filter sizes, learning rate) will be tuned on validation only.
- **Status:** Changed. Both models were built and compared; the plain CNN was kept (see D-10).

### D-9: CNN input sampling rate — 100 Hz, decided by experiment
- **Date:** 04 Oct 2026
- **Context:** D-2 chose 500 Hz for hand-crafted features. After D-6 made a CNN the main ECG model, the first plain CNN at 500 Hz took about 1.5 hours to train on CPU, too slow for the ResNet and fusion experiments still to come.
- **Options considered:**
  - **Option A: Stay at 500 Hz on CPU**
    - Pros: no change.
    - Cons: an estimated 2–3 hours per ResNet run, so very few experiments are possible.
  - **Option B: Stay at 500 Hz, train on Google Colab GPU**
    - Pros: faster training, same data.
    - Cons: 2.6 GB upload, free-tier session limits and disconnects, a second environment to maintain.
  - **Option C: Test 100 Hz on the same plain CNN, with a rule fixed before running**
    - Rule: switch if 100 Hz validation AUROC ≥ 0.928 (within 0.01 of the 500 Hz result, 0.938).
    - Pros: decided by our own evidence.
    - Cons: one extra run. The same 7-sample filter spans 70 ms instead of 14 ms, so this compares the same architecture at two rates rather than isolating the sampling rate.
- **Evidence** (train_cnn.py, seed 42, identical settings except `downsample`; 100 Hz made by averaging every 5 consecutive 500 Hz samples):

  | | 500 Hz | 100 Hz |
  |---|---|---|
  | Best val AUROC | 0.938 (epoch 19) | 0.940 (epoch 11) |
  | Best val AUPRC | 0.957 | 0.958 |
  | Sensitivity at 90% specificity | 0.825 | 0.829 |
  | Time per epoch (CPU) | 4.2–6.5 min | 1.2–1.6 min |
  | Stopping | Hit the 20-epoch cap | Early-stopped at epoch 15 |

- **Decision:** Option C was run. 0.940 ≥ 0.928, so all further CNN work uses 100 Hz. Decided by me, applying the rule set in advance.
- **Reasoning:** Equal accuracy at about one-fifth of the training time allows more experiments in the time available. We do **not** claim 100 Hz is better: a 0.002 difference from a single seed is within run-to-run noise.
- **Risks and limitations:** Very fine waveform detail is lost, although this experiment showed no measurable cost. Single-seed comparison.
- **Status:** Active. D-2's sampling-rate choice is marked Changed.

### D-10: ECG model — keep the plain CNN; ResNet did not clear the bar
- **Date:** 04 Oct 2026
- **Context:** D-7 planned a plain CNN first, then a 1D ResNet. Before training the ResNet we fixed a rule: adopt the ResNet only if its validation AUROC is at least 0.945 (0.005 above the plain CNN's 0.940 at 100 Hz); otherwise keep the plain CNN, because it is simpler and a smaller gain is within run-to-run noise.
- **Evidence** (train_cnn.py, seed 42, 100 Hz, identical data, loop and metrics):

  | | Plain CNN | 1D ResNet |
  |---|---|---|
  | Trainable parameters | 189,889 | 1,023,233 |
  | Best val AUROC | 0.940 (epoch 11) | 0.941 (epoch 9) |
  | Best val AUPRC | 0.958 | 0.960 |
  | Sensitivity at 90% specificity | 0.829 | 0.834 |
  | Time per epoch (CPU) | 1.2–1.6 min | 3.3–4.5 min |
  | Stopping | Early-stopped at epoch 15 | Early-stopped at epoch 13 |

- **Options considered:**
  - **Option A: Plain CNN**
    - Pros: 5x fewer parameters, about 3x faster, simpler to explain; same accuracy within noise.
    - Cons: lower capacity if later work needs it.
  - **Option B: 1D ResNet**
    - Pros: marginally higher on all three metrics.
    - Cons: the gain (+0.001 AUROC) is far below the pre-agreed 0.005 bar and within single-seed noise; slower and harder to explain.
- **Decision:** Option A, plain CNN at 100 Hz. Decided by me, applying the rule set before the run.
- **Reasoning:** On this dataset and task, skip connections did not add measurable value at this depth. Both models appear to hit the same ceiling (validation AUROC about 0.94), which suggests the limit is the binary label and data rather than the architecture.
- **Risks and limitations:** Single seed per model. The ResNet weights are kept (outputs/models/resnet1d_100hz.pt) and the comparison is reported as an ablation.
- **Status:** Active

### D-11: Fusion design — cross-fitted stacking
- **Date:** 04 Oct 2026
- **Context:** The challenge requires a model that ingests both the static data (age, sex) and the ECG. We needed a fusion design that is leak-free and lets the dashboard show how much each input contributed.
- **Options considered:**
  - **Option A: Simple stacking** (logistic regression on the single CNN's probability + age + sex, fitted on the training set)
    - Pros: tiny and transparent.
    - Cons: leaks. The CNN has seen the training ECGs, so its training-set probabilities are overconfident, and the regression learns to over-trust it.
    - Time cost: about 1 hour.
  - **Option B: Cross-fitted stacking**: training set split into 4 patient-safe groups by official fold pairs (1+2, 3+4, 5+6, 7+8); a plain CNN is trained 4 times, each scoring the group it never saw; a logistic regression is fitted on [ECG log-odds, age per decade, sex]. Validation ECG scores are the average of the 4 CNNs.
    - Pros: leak-free. Exact, signed contributions for the dashboard ("each extra decade multiplies the odds by X, same ECG"). Coefficients are easy to defend to a jury.
    - Cons: 4 extra CNN trainings. Averaging 4 CNNs is itself an ensemble, which can lift scores independently of age and sex, so the fair comparison must be against the ECG-only ensemble, not the single CNN. Cannot learn age-by-ECG interactions.
    - Time cost: about 1 hour of machine time.
  - **Option C: End-to-end fusion** (age and sex concatenated to the CNN embedding, trained in one run)
    - Pros: leak-free, one ~20-minute run, can learn interactions.
    - Cons: contribution of age and sex only visible through ablation and what-if; 2 inputs next to 128 may be largely ignored.
    - Time cost: about 2 hours including code.
- **Decision:** Option B. Decided by me, on mentor recommendation (the mentor first recommended C on cost, then changed to B once I said compute time was not a constraint).
- **Reasoning:** Explainability of the static contribution matters for the doctor dashboard and the jury, and B gives it exactly without leakage. We gave up interaction learning.
- **Implementation details:**
  - Fold models train for a fixed 11 epochs (the best epoch of the single plain CNN at 100 Hz), without early stopping, so neither the left-out group nor the validation set influences model selection.
  - Seeds 42–45, one per fold model. The test set is not scored; the 4 fold models are saved for the single final test evaluation.
  - Age enters as (age − 60) / 10, so its coefficient is per decade.
- **Pre-agreed evaluation rule (set before running):** fusion "adds measurable value" only if the 95% paired-bootstrap confidence interval of AUROC(fused) − AUROC(ECG-only 4-CNN ensemble) on validation lies entirely above 0. Otherwise it is reported as "no measurable difference", and the fused model is still used in the dashboard for its explanations.
- **Risks and limitations:** Training-set ECG scores come from single CNNs trained on 75% of the data, while validation scores are a 4-model average; their distributions differ slightly, which can distort the regression's weight on the ECG score. Age and sex are thin static data (L-4), so a small or zero gain is expected.
- **Results, stage 1** (crossfit_cnn.py, 04 Oct 2026): patients spanning more than one group: 0. Group sizes A 4,253 · B 4,255 · C 4,229 · D 4,236.

  | Fold model | AUROC on its left-out group | AUROC on validation | Time |
  |---|---|---|---|
  | A | 0.949 | 0.936 | 13.9 min |
  | B | 0.939 | 0.930 | 12.2 min |
  | C | 0.949 | 0.941 | 10.6 min |
  | D | 0.940 | 0.925 | 10.7 min |

  OOF AUROC over all 16,973 training ECGs: 0.941. Validation AUROC of the 4-CNN average: 0.942.
  **Observation:** identical models with different seeds and 75% subsets range 0.925–0.941 on validation (spread 0.016). Single-model run-to-run noise is therefore larger than the 0.005 margins used in D-9 and D-10, which supports treating those small differences as noise.
- **Results, stage 2** (fuse.py, validation set, 04 Oct 2026):

  | Model | AUROC | AUPRC | Sensitivity at 90% specificity |
  |---|---|---|---|
  | Static only (age + sex) | 0.755 | 0.799 | 0.386 |
  | ECG only (4-CNN ensemble) | 0.942 | 0.960 | 0.835 |
  | Fused (ECG score + age + sex) | 0.941 | 0.960 | 0.823 |
  | Reference: single plain CNN | 0.940 | 0.958 | 0.829 |

  Chance AUPRC (abnormal share): 0.576.
  Fusion gain over ECG-only ensemble: −0.0007 AUROC, 95% paired-bootstrap CI −0.0025 to +0.0010 (2,000 resamples). **Pre-agreed rule: no measurable difference.**
  Coefficients, fused model: ECG log-odds +0.819; age +0.175 per decade (odds ×1.19, same ECG); sex (1 vs 0) −0.214 (odds ×0.81).
  Coefficients, static-only model: age +0.531 per decade (odds ×1.70); sex −0.778 (odds ×0.46).
  **Interpretation:** age and sex carry real signal on their own (0.755 AUROC), but most of it is already contained in the ECG: their coefficients shrink by roughly two-thirds once the ECG score is present. The narrow confidence interval means fusion cannot be adding more than about +0.001 AUROC on this data. The ECG coefficient below 1 means the regression shrinks the CNN's confidence, a calibration effect to be examined in the evaluation phase. The drop in sensitivity (0.835 → 0.823) corresponds to about 15 of 1,231 abnormal validation patients and is not tested for significance.
- **Status:** Active

### D-12: Final risk score — fused model or ECG-only ensemble
- **Date:** 04 Oct 2026
- **Context:** D-11 showed the fused model and the ECG-only 4-CNN ensemble rank patients equally well on validation (AUROC 0.941 vs 0.942, no measurable difference). The dashboard shows the score to a doctor as a probability, so we needed to choose which model's output is shown.
- **Options considered:**
  - **Option A: ECG-only ensemble**
    - Pros: simplest; marginally higher sensitivity at 90% specificity (0.835 vs 0.823, within noise).
    - Cons: no way to show how age and sex contribute; CNN overconfidence (fused ECG coefficient 0.819 < 1) uncorrected.
  - **Option B: Fused model**
    - Pros: same accuracy, plus exact signed contributions and a what-if control for the dashboard; may be better calibrated.
    - Cons: keeps inputs that add no measurable accuracy (L-6); one more component to explain.
  - **Option C: Decide by a calibration check on validation, with the rule fixed in advance**
    - Pros: evidence-based, consistent with D-9 and D-10.
    - Cons: a short delay before the dashboard design is fixed.
- **Decision:** Option C. Decided by me, on mentor recommendation.
- **Pre-agreed rule (set before running calibration_check.py):** the fused model becomes the final risk score unless its validation Brier score is higher (worse) than the ECG-only ensemble's. ECE, log loss, a reliability diagram and a bootstrap interval for the Brier difference are reported for information.
- **Results** (calibration_check.py, validation, 2,137 patients, actual abnormal rate 0.576):

  | Model | Brier | ECE | Log loss | Mean predicted |
  |---|---|---|---|---|
  | Static only (age + sex) | 0.1983 | 0.0277 | 0.5804 | 0.584 |
  | ECG only (4-CNN ensemble) | **0.0963** | 0.0287 | 0.3052 | 0.547 |
  | Fused (ECG + age + sex) | 0.0970 | 0.0290 | 0.3068 | 0.554 |

  Brier difference fused − ECG: +0.00066, 95% bootstrap CI −0.00092 to +0.00224 (no measurable difference). **Under the pre-agreed rule the fused Brier is higher, so the final risk score is the ECG-only ensemble.**
- **Observation:** the ensemble is mildly **under**confident, not overconfident as the mentor had expected: in 9 of 10 risk bins the actual abnormal rate is above the predicted rate (gaps +0.01 to +0.08), and its mean prediction (0.547) is below the actual rate (0.576). A likely cause is that averaging four models pulls probabilities towards the middle. This also explains the fused model's ECG weight of 0.819: it was fitted on single-model out-of-fold scores, which are more extreme than the 4-model average it receives on validation (the distribution mismatch flagged in D-11).
- **Consequences:** age and sex are not inputs to the final risk score. On the dashboard they are shown as patient context, not as model reasoning. The fused model's coefficients remain a reported analysis result (D-11). Recalibrating the ensemble is a possible later decision; it would need data not used for evaluation.
- **Status:** Active

### D-13: Wearable stream — hybrid simulation
- **Date:** 04 Oct 2026
- **Context:** We have no wearable device and no open dataset links wearables to PTB-XL patients. The twin needs a daily series per patient (resting heart rate, HRV, sleep, steps) for the personal-baseline monitor.
- **Rules common to all options:** (1) the simulated series must not depend on the patient's ECG label, to avoid circular "fusion"; (2) deterioration episodes are injected deliberately in a labelled subset, and the monitor is evaluated only on detecting those, presented as a synthetic test, never as a clinical result.
- **Options considered:**
  - **Option A: Pure rule-based generator** with ranges taken from published literature.
    - Pros: fast, fully controlled.
    - Cons: every number needs a literature citation; realism depends on those choices.
    - Time cost: about half a day.
  - **Option B: Resample real weeks from an open wearable dataset** and assign them to PTB-XL patients.
    - Pros: most realistic noise and habits.
    - Cons: real people's data paired with unrelated ECG patients; download, format and licence work.
    - Time cost: about 1–1.5 days.
  - **Option C: Hybrid** — estimate generator parameters (typical values, day-to-day variability, weekday effects) from an open dataset, then simulate.
    - Pros: sourced, realistic parameters; the final product contains no real person's wearable data.
    - Cons: still one download and a licence check.
    - Time cost: about 1 day.
- **Decision:** Option C. Decided by me, on mentor recommendation.
- **Reasoning:** gives a sourced answer to "where did these numbers come from?" while keeping the product fully synthetic.
- **Risks and limitations:** parameters come from a small, non-Indian, non-cardiac population (see D-14); the monitor's detection results measure performance on our own injected scenarios only.
- **Status:** Active

### D-14: Sources for the wearable simulation recipe
- **Date:** 05 Oct 2026
- **Context:** D-13 chose a hybrid simulation: generator parameters estimated from open data, then fresh synthetic series generated per patient. No real person's wearable data is assigned to any PTB-XL patient. I questioned whether about a dozen people is enough. The answer: within-person day-to-day dynamics need long series (available), while the between-person spread needs many people, which is better taken from published population studies.
- **Facts checked on official pages (04–05 Oct 2026):**
  - PMData (datasets.simula.no/pmdata): ~~CC BY 4.0; 12 participants; 3 months~~ **CORRECTED 05 Oct 2026, see below**; daily resting heart rate, sleep and steps files; no HRV listed; hosted on OSF.
  - MMASH (physionet.org/content/mmash): Open Access, Open Data Commons ODbL v1.0; 22 healthy young adult males; 24 hours each; beat-to-beat heart data (allows HRV), sleep, actigraphy/steps; 23.5 MB.
  - Crowd-sourced Fitbit datasets (zenodo.org/record/53894): marked Open; 30 users; minute-level heart rate, activity, sleep; **no licence shown** in what was retrieved.
  - Wearable Activity Tracker Data (zenodo.org/records/7621224): restricted, application required; 88 users aged 18–31; 4 months; 267.7 GB.
  - LifeSnaps (zenodo.org/records/6832186): files shown as restricted; licence not stated.
- **Options considered:**
  - **Option A: PMData (day-to-day dynamics) + MMASH (HRV) + cited population studies (between-person ranges)**
    - Pros: about 34 real people, all from verified open sources; covers HRV; every number traceable.
    - Cons: small, young, healthy, non-Indian, non-cardiac source populations; MMASH's ODbL share-alike terms to be read.
    - Time cost: about 1.5 days.
  - **Option B: A plus the 30-user crowd-sourced Fitbit data**
    - Pros: about 64 people; better estimate of how much people differ in day-to-day variability.
    - Cons: licence not confirmed; mixed devices, noisier data.
    - Time cost: about 2 days.
  - **Option C: Apply for the 88-user restricted dataset**
    - Cons: approval time unknown, 267.7 GB, ages 18–31 only. Not feasible before the deadline.
- **Decision:** Option A. Recommended by the mentor; I delegated the choice ("you suggest") and approved it.
- **Reasoning:** the only option with every licence verified; it fills the HRV gap; and the extra people in B mainly refine one recipe parameter, at the cost of an unconfirmed licence.
- **Reversibility:** if the Fitbit dataset's licence is later confirmed open, it can be added as a third source without changing the design.
- **Risks and limitations:** see D-13. To do: read the ODbL v1.0 terms for MMASH; verify each population-range citation before use.
- **Status:** Active

#### D-14 correction (05 Oct 2026) — PMData facts checked against the downloaded files
The figures above came from an automated summary of the Simula web page. Checking the downloaded files directly showed two errors:
- **Licence:** the `license.txt` file shipped with the download states **Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)**, not CC BY 4.0. The NonCommercial condition is a new constraint (see D-14a).
- **Participants:** `participant-overview.xlsx` lists **16** participants (p01–p16), aged 23–60 (13 male, 3 female), not 12.
- **Usable resting heart rate:** 14 of 16 participants have `resting_heart_rate.json` (p12 and p13 do not). After removing zero values, **1,658 person-days** remain; per-person series run 35–152 days. Per-person median resting HR ranges 47.0–70.4 bpm; the standard deviation of day-to-day changes ranges 0.81–1.36 bpm. (Computed by the mentor from the staged files; to be reproduced by `explore_wearables.py`.)
- **MMASH files confirmed:** 22 users, each with `RR.csv` (beat-to-beat intervals, column `ibi_s`), `Actigraph.csv` (per-second HR and steps), `sleep.csv`, `user_info.csv`. The first RR intervals of user_1 include implausibly short values (e.g. 0.258 s), so artifact filtering is required before computing HRV.

### D-14a: Keep PMData despite its NonCommercial licence
- **Date:** 05 Oct 2026
- **Context:** the downloaded PMData `license.txt` states CC BY-NC 4.0 (see D-14 correction). The challenge offers cash prizes and an internship and is run by a company, so whether this use counts as "non-commercial" is a legal question the mentor could not answer.
- **Options considered:**
  - **Option A: Keep PMData and declare the licence openly** — use it only to compute summary parameters; never redistribute its files in the repo; attribute it and state CC BY-NC 4.0 in the README.
    - Pros: keeps the only real multi-week source of day-to-day behaviour (MMASH covers 24 hours per person).
    - Cons: residual risk that organisers or judges consider the NC licence incompatible with a prize competition.
  - **Option B: Drop PMData** — MMASH plus literature only.
    - Pros: no NC question.
    - Cons: no real data at all for day-to-day dynamics, the core of the monitor.
  - **Option C: Ask the organisers, build with PMData meanwhile** (mentor's recommendation).
    - Pros: an official answer; a "no" costs one recipe-file swap.
    - Cons: a reply may not arrive before the deadline.
- **Decision:** use PMData (Option A). Decided by me. The mentor's suggestion to email the organisers remains open as an optional extra step.
- **Reasoning:** PMData is the only source of real multi-week day-to-day behaviour; the code keeps it isolated so it can be swapped out if needed.
- **Risks and limitations:** the NC licence must appear in the README's licence section with attribution; raw PMData files are excluded from the public repo; only derived summary parameters are published. I am responsible for confirming the licence is acceptable for my use.
- **Status:** Active

- **05 Oct 2026, explore_wearables.py written and dry-run by the mentor** on the files staged from my computer (all 14 PMData resting HR files, p01's sleep and steps, MMASH user_1). Findings so far: PMData `sleep.json` contains exact duplicate records (same `logId`; 5 removed for p01), now de-duplicated; MMASH user_1 night RMSSD 75.2 ms after removing 3.3% of beats as artifacts. Full results to come from my run on all files.

- **05 Oct 2026, first full run of explore_wearables.py (on my computer)**:
  - PMData: 16 participants; valid days — resting HR 1,658 (14 people), sleep 1,881 (16 people), steps 1,488 (16 people, days with at least 600 minutes of step records). Sleep duplicates removed: 2–6 per participant. Per-person day-to-day SD of resting HR: 0.81–1.38 bpm. Some participants have few valid step days (p03 and p16: 33 days each; p16 median 2,489 steps/day), consistent with the tracker not being worn all day; to be handled when fitting the recipe.
  - MMASH, first run: 15 of 22 users gave a night RMSSD (median 62.3 ms, range 33.9–103.5). 6 users failed and 1 had no sleep record.
  - **Bug found and fixed:** for the 6 failing users, MMASH's `sleep.csv` labels the out-of-bed time with the same day number as going to bed (e.g. in bed day 1 22:29, out of bed "day 1" 05:52), so the night window was empty. Fix: an end time earlier than the start time is moved to the next day.
  - **Quality rule added (mentor's choice, open to my override):** a night where more than 20% of beats are removed as artifacts is excluded from HRV. This excludes user_4 (29.9% removed; its RMSSD of 103.5 ms was the highest value and likely inflated by noise).
  - user_18's age is recorded as 0 in `user_info.csv`; set to unknown.
  - **Second run:** 18 of 22 MMASH users gave a valid night RMSSD (median 57.4 ms, range 33.9–86.6 ms). Remaining failures: user_4 (excluded, 30% artifacts), user_11 (no sleep record), users 8 and 9 (see below).
  - **Third run (after both fixes): 20 of 22 MMASH users have a valid night RMSSD**, median 57.4 ms, range 33.9–86.6 ms (from outputs/wearables/summary.txt). Excluded: user_4 (artifacts), user_11 (no sleep record).
  - **Second quirk found and fixed:** users 8 and 9 code their second recording day as "-29" instead of 2 (day 1 runs ~09:50–23:59; the "-29" rows run 00:00–09:20). Day numbers below 1 are now treated as day 2. Mentor's dry run on the staged files: user_8 RMSSD 69.8 ms (6.8% artifacts), user_9 46.2 ms (0.4%). Third run pending.

### D-15: How each simulated patient's personal wearable baseline is set
- **Date:** 05 Oct 2026
- **Context:** the simulator needs a personal "normal" level per patient for resting HR, HRV, sleep and steps; day-to-day wobble then happens around it. PMData participants are fit (per-person median resting HR 47.0–70.4 bpm), while our PTB-XL patients are older hospital patients (median age 62), so copying PMData levels would be unrealistic.
- **Options considered:**
  - **Option A: draw levels directly from PMData/MMASH people** — simple, but gives older cardiac patients an athlete's physiology.
  - **Option B: levels from published population studies by age and sex; PMData supplies only the day-to-day wobble** — realistic and sourced; needs verified literature.
  - **Option C: resting HR anchored to the patient's own PTB-XL ECG heart rate** — genuinely personal; needs a cited clinic-to-home adjustment and about a day more. Uses an ECG *measurement*, not the label; acceptable because the monitor works on deviations from each patient's own baseline, not on the baseline level, which must be stated explicitly.
- **Decision:** Option B now; Option C to be added afterwards as an extension, because I want the personalisation it gives. Decided by me, on mentor recommendation.
- **Sources found so far** (checked by the mentor on 05 Oct 2026; see the table in the conversation and the list below):
  - **Resting HR level:** Quer et al. 2020, PLOS ONE (doi:10.1371/journal.pone.0227709), CC BY. 92,457 US adults, Fitbit. Mean daily RHR 65.5 ± 7.7 bpm (Results); women higher in all age brackets (Results); 95% of women 53–82 bpm, 95% of men 50–80 bpm (Results); average RHR rises until about age 50 then declines (Results, Fig. 3A — no table of values found).
  - **Steps level:** Althoff et al. 2017, Nature. 717,527 smartphone users, 111 countries. Average 4,961 steps/day, SD 2,684 (Methods); **India average 4,297 steps/day (Supplementary Table 1)**; activity decreases with age and is lower in women (Extended Data Fig. 1 caption).
  - **HRV vs age:** Natarajan et al. 2020, Lancet Digital Health (8.2 million Fitbit users; CC BY-NC-ND 4.0): HRV metrics decrease with age (abstract; no age values in the abstract). Umetani et al. 1998, JACC (260 healthy subjects aged 10–99, 24-h Holter; paywalled): rMSSD and pNN50 "decreased most rapidly, reaching 24% and 47% of baseline, respectively, by the sixth decade". **Which percentage belongs to which measure must be verified in the full text before use.**
  - **Sleep vs age:** Ohayon et al. 2004, Sleep 27(7):1255–73 (meta-analysis). **Values not yet retrieved**; until verified, sleep levels come from PMData.
- **Risks and limitations:** US/global smartphone and Fitbit populations, not Indian cardiac patients (except the India steps figure). Where a source gives only a direction (e.g. "decreases with age") and no number, the simulator will not invent an age slope.
- **Status:** Active

- **05 Oct 2026, fit_recipe.py run by the mentor on my real processed files** (pmdata_daily.csv, mmash_hrv.csv staged from my computer). Rule: a person contributes day-to-day dynamics only with at least 60 valid days. Model per person and measure: level + weekday effect + AR(1) wobble (phi = lag-1 autocorrelation on consecutive days, sd = residual SD); steps on a log scale.

  | Measure | People used | Memory phi (median, range) | Wobble sd (median, range) | Largest weekday effect |
  |---|---|---|---|---|
  | Resting HR | 13 (p04, p12, p13 excluded) | 0.86 (0.64–0.94) | 2.04 bpm (1.28–3.35) | Sun +0.36 bpm |
  | Sleep | 14 (p12, p13 excluded) | 0.04 (−0.20–0.16) | 1.04 h (0.72–1.94) | Sat +0.26 h |
  | Steps (log) | 12 (p03, p13, p14, p16 excluded) | 0.14 (−0.14–0.37) | 0.318 log units (~32%) | Sun −0.086 (~ −8%) |

  - Consistency check: with sd 2.04 and phi 0.86, the implied SD of day-to-day changes is 2.04 × √(2 × (1 − 0.86)) ≈ 1.08 bpm, matching the directly measured 0.81–1.38 bpm in explore_wearables.py.
  - **Implication for the monitor (to carry into D-17):** resting HR has strong memory (phi 0.86), so healthy people drift slowly for days to weeks. A monitor comparing each day with a fixed long baseline will raise false alarms on these natural drifts.
  - Personal levels: resting HR women 67.5 ± 7.40, men 65.0 ± 7.65 bpm (derived from Quer et al. 2020); steps log-normal with mean 4,297 and SD 2,684 (Althoff et al. 2017, India); sleep 6.78 ± 0.57 h (PMData placeholder); HRV median 57.4 ms, log-SD 0.278 (MMASH, 20 users).
  - **Open gap:** HRV day-to-day dynamics cannot be measured from our data (MMASH has one night per person). Marked UNSOURCED in recipe.json; see D-15a.

### D-15a: Filling the HRV day-to-day gap
- **Date:** 05 Oct 2026
- **Context:** MMASH has one night per person, so our data cannot show how much one person's HRV changes from night to night. The simulator needs that number.
- **Options considered:**
  - **Option A: drop HRV from the wearable stream** — honest, but HRV is one of the challenge's example measures.
  - **Option B: make HRV move with resting HR using the between-person relationship in MMASH** — rejected: using a between-person relationship to model within-person change is a statistical error.
  - **Option C: take night-to-night HRV variability from a published study**, with A as fallback.
- **Decision:** Option C. Decided by me, on mentor recommendation.
- **Source found (read by the mentor in the built-in browser on 05 Oct 2026):** Hannon J, O'Hagan A, Lambe R, O'Grady B, Doherty C. *Associations Between Daily Heart Rate Variability and Self-Reported Wellness: A 14-Day Observational Study in Healthy Adults.* Sensors 2025;25(14):4415, doi:10.3390/s25144415 (PMC12300306). 41 healthy adults, daily 5-minute morning recordings with a Polar H10 chest strap (Abstract; Section 2). Section 4 (Discussion): "The average CV was 0.37 (SD = 0.16), with individual values ranging from 0.14 to 0.71."
- **How it is used:** HRV wobble on the log scale = sqrt(ln(1 + 0.37²)) = 0.358.
- **Remaining assumptions (labelled ASSUMPTION in recipe.json):**
  - HRV day-to-day memory (phi) has no source; it is set equal to the measured resting HR memory (0.861). This is the conservative choice: stronger memory produces longer natural drifts, which makes the monitor's job harder, not easier.
  - No weekday effect for HRV.
  - Between-person spread of HRV wobble not estimated (the paper reports only mean, SD and range of CV).
- **Risks and limitations:** the source measured morning 5-minute readings, while our HRV levels come from night-time recordings (MMASH). Healthy adults, not cardiac patients.
- **Status:** Active

### D-16: What simulated deterioration looks like
- **Date:** 05 Oct 2026
- **Context:** to test the wearable monitor, some simulated patients need injected deterioration episodes. No verified source describes what cardiac deterioration looks like on a consumer wearable.
- **Options considered:**
  - **Option A: a grid of episode sizes, reported as detection limits** ("how big a change can the monitor catch, and how fast?").
    - Pros: claims no clinical knowledge we do not have; gives an honest detection-rate-vs-size chart.
    - Cons: no single headline accuracy number.
  - **Option B: one shape from published wearable studies of infection onset (flu/COVID).**
    - Pros: a real, sourced pattern. Cons: infection is not cardiac deterioration.
  - **Option C: one shape from heart-failure remote-monitoring studies.**
    - Pros: clinically closest. Cons: no source found or verified yet; such studies often use implanted devices.
- **Decision:** Option A, with one Option-B-style example to be shown alongside for context (clearly labelled as an infection pattern) once a source is verified. Decided by me, on mentor recommendation.
- **Implementation (simulate_wearables.py, seed 2026):**
  - Every validation and test patient gets 60 synthetic days of resting HR, night HRV, sleep and steps from recipe.json. Only age and sex are read; the ECG label is never used.
  - Half the patients, chosen at random, get one episode. Resting HR rises by +3, +5, +8 or +12 bpm (severity 1–4), building linearly over 3, 5 or 7 days from a random start day between 36 and 53, then holding.
  - Other measures worsen at the same time by fixed, modest amounts per severity (scenario design choices, not clinical facts): HRV −10/−15/−20/−30%, sleep −0.25/−0.5/−0.75/−1.0 h, steps −10/−20/−30/−40%.
  - **Changed during build:** the first version moved the other measures by the same number of their own wobble-SDs as resting HR. That produced implausible episodes (one example patient's sleep fell from about 7 h to 1–2 h, HRV from ~110 to ~15 ms), because sleep and steps are much noisier than resting HR. Replaced with the fixed amounts above.
- **Results (mentor's run on my staged files):**
  - Validation: 2,137 patients, 1,066 with an episode (49.9%); test: 2,145 patients, 1,071 (49.9%). Severity groups 252–287 patients each.
  - Independence check: episode share 0.490 among normal-ECG patients vs 0.505 among abnormal-ECG patients.
  - Simulated levels before any episode (validation): resting HR median 66.5 bpm, HRV 57.1 ms, sleep 6.79 h, steps 3,597.
  - Realism check vs PMData (clean simulated patients): SD of day-to-day change — resting HR 1.18 real vs 1.21 simulated; sleep 1.82 vs 1.59 h; log steps 0.50 vs 0.44. Memory — resting HR 0.86 vs 0.80; sleep 0.01 vs 0.01; steps 0.14 vs 0.11. This is a consistency check (the recipe was fitted on PMData), not independent validation. Simulated sleep and steps are slightly calmer than the real data.
  - **Observation:** a simulated patient without any episode shows a natural resting HR drift of about +7 bpm over ~10 days (example_patients.png), as large as a severity-3 episode. Natural drifts of this size also appear in the real PMData (e.g. p05, p14), so the monitor must cope with them.
- **Risks and limitations:** episode shapes and co-movement amounts are design choices; detection results describe performance on these scenarios only. No missing days are simulated, although real wearable data has gaps.
- **Status:** Active

### D-17: The wearable monitor rule
- **Date:** 05 Oct 2026
- **Context:** the monitor must flag deterioration while ignoring natural drift, which D-16 showed can be as large as a +8 bpm episode in resting HR.
- **Options considered:**
  - **Option A: fixed personal baseline (days 0–27) + 3-day average score above a threshold.** Simplest; expected to false-alarm on natural drift.
  - **Option B: rolling personal baseline (the 28 days ending 8 days ago) + 3-day average score + k consecutive days above threshold.** Designed for drift; more settings.
  - **Option C: learned anomaly detector (Isolation Forest).** No hand-picked thresholds, but hard to explain on a doctor's dashboard.
- **Decision:** Option B, with Option A run alongside as the comparison. Decided by me, on mentor recommendation.
- **Implementation (monitor.py):** for each measure, z = direction × (value − baseline median) / (1.4826 × MAD); resting HR up, HRV/sleep/steps down count as unhealthy; HRV and steps on a log scale; daily score = mean of the four z values, smoothed over 3 days. Monitored days 35–59. A false alarm is an alarm event starting on a non-episode day; rates are per patient-month (30 days).
- **Tuning rule (fixed before results, mentor's choice, open to my override):** on validation patients only, pick the settings with the highest average detection across the four severities, subject to at most **0.5 false alarms per patient-month**. Settings are saved to outputs/monitor/monitor_config.json and locked before the test patients are scored. The first grid had the rolling method's best threshold at its lower edge (0.5), so the grid was widened (thresholds 0.0–4.0 in steps of 0.05; persistence 1–6) before choosing.
- **Validation results (synthetic data, synthetic episodes; mentor's run on my staged files):**

  | Method | Settings | False alarms / patient-month | +3 bpm | +5 bpm | +8 bpm | +12 bpm |
  |---|---|---|---|---|---|---|
  | A: fixed | threshold 1.10, k = 1 | 0.47 | 65.5% (5 d) | 86.4% (4 d) | 97.7% (3 d) | 100.0% (2 d) |
  | B: rolling | threshold 0.50, k = 4 | 0.50 | 74.5% (5 d) | 90.2% (5 d) | 95.8% (5 d) | 99.6% (4 d) |

  Cells show detection rate and median delay from episode start to first flag. Average detection: B 90.0% vs A 87.4%.
- **Interpretation:** at the same false-alarm budget, B catches more small episodes (+9 percentage points at +3 bpm, about 255 episodes per group, so this gap is only moderately larger than sampling noise) but flags large episodes about 2 days later, because it waits for 4 consecutive days. A performs better than expected because averaging four measures dilutes resting HR drift. In a real deployment longer than 60 days a fixed baseline would go stale, which this test does not capture.
- **Risks and limitations:** results apply only to our synthetic episodes; equal weighting of the four measures was not tuned; no missing days.
- **Status:** Active

### D-18: The twin-state algorithm (fusion of the two streams)
- **Date:** 05 Oct 2026
- **Context:** the official guidelines require "a working algorithmic model that ingests these two streams". Since D-12 the risk score uses the ECG only, so the combination with the wearable stream must itself be an algorithm.
- **Options considered:**
  - **Option A: rule table only** (ECG risk tier × monitor flag → Stable/Watch/Escalate). Transparent, but the streams meet only at the end.
  - **Option B: risk-adjusted monitoring plus the same table** — the ECG risk tier also sets how sensitive the wearable monitor is. Real fusion; but our synthetic episodes are independent of the ECG, so the data cannot show that outcomes improve, only how alerts are redistributed.
  - **Option C: one combined probability** — would require an invented formula linking wearable deviations to cardiac events; no data supports it.
- **Decision:** Option B. Decided by me, on mentor recommendation.
- **Implementation (twin_state.py), rules fixed before results:**
  - Step 1: high risk if P(abnormal ECG) ≥ the 90%-specificity threshold on validation (0.511).
  - Step 2: the overall false-alarm budget stays 0.5 per patient-month, but a high-risk patient may receive twice as many false alarms as a low-risk one (ratio 2 is a design choice). Each tier's monitor settings are tuned on validation within its own budget.
  - Step 3: daily state — low risk & no flag = STABLE; low risk & flag or high risk & no flag = WATCH; high risk & flag = ESCALATE.
- **Validation results (ECG risk real; wearables and episodes synthetic):**
  - 1,118 high-risk (52.3%), 1,019 low-risk. Budgets: low 0.33, high 0.66 false alarms per patient-month. Settings: low-risk threshold 0.70, k = 4; high-risk threshold 0.55, k = 3.

  | Group | Monitor | False alarms / month | +3 | +5 | +8 | +12 | Mean |
  |---|---|---|---|---|---|---|---|
  | High risk | uniform (D-17) | 0.47 | 72.3% | 92.7% | 95.7% | 100.0% | 90.2% |
  | High risk | risk-adjusted | 0.61 | 77.7% | 94.7% | 97.1% | 100.0% | 92.4% |
  | Low risk | uniform (D-17) | 0.53 | 76.8% | 87.5% | 95.9% | 99.3% | 89.9% |
  | Low risk | risk-adjusted | 0.30 | 63.2% | 80.9% | 95.9% | 99.3% | 84.8% |
  | All | uniform (D-17) | 0.50 | 74.5% | 90.2% | 95.8% | 99.6% | 90.0% |
  | All | risk-adjusted | 0.47 | 70.6% | 88.2% | 96.5% | 99.6% | 88.7% |

  - Worst state per patient over days 35–59: STABLE 446 (20.9%), WATCH 931 (43.6%), ESCALATE 760 (35.6%). Of ESCALATE patients, 68.3% had an injected episode and 93.0% had an abnormal ECG.
- **Interpretation:** risk adjustment moves sensitivity from low-risk to high-risk patients (+2.2 points mean detection for high risk, −5.1 for low risk; −1.3 overall at a similar total false-alarm rate). This is the intended trade-off, not an improvement in overall accuracy.
- **Risks and limitations:**
  - Episodes are independent of the ECG by design, so no outcome benefit can be shown (stated in the code and outputs).
  - Because 52% of this hospital population is high-risk (L-5), every high-risk patient starts at WATCH; in a screening population the high-risk share, and so the WATCH share, would be much lower.
  - 42% of high-risk patients without an episode reach ESCALATE at least once in 25 days; this is the cost of the higher false-alarm budget for that tier.
- **Status:** Active

- **05 Oct 2026, reproducibility check passed.** I ran fit_recipe.py, simulate_wearables.py, monitor.py and twin_state.py on my own computer. Every printed number matched the mentor's runs exactly (e.g. 1,066 validation episodes; rolling monitor 74.5/90.2/95.8/99.6%; twin risk-adjusted mean 88.7%), confirming the fixed seeds make the wearable pipeline reproducible.

- **05 Oct 2026, settings locked by me** for the final test run: false-alarm limit 0.5 per patient-month (D-17) and high-risk allowance ×2 (D-18), after comparing the alternatives on validation (limit 0.25 / 0.5 / 1.0 gave mean detection 83.7% / 90.0% / 96.3%; ratio ×1 / ×2 / ×3 gave low-risk +3 bpm detection 76.0% / 63.2% / 56.8% and high-risk 72.3% / 77.7% / 83.1%).

- **05 Oct 2026, PROTOCOL DEVIATION (disclosed by the mentor):** while dry-running `final_evaluation.py` to check it for bugs, the mentor replaced the ECG model with fake random scores, but Parts 2–3 ran on the real **test-split synthetic wearable data**. So the wearable monitor's test results were seen before the official run (rolling monitor on test: 0.51 false alarms/month; detection 76.2% / 92.9% / 97.4% / 99.3% for +3/+5/+8/+12 bpm). The ECG test results were **not** seen (fake scores only), and the twin results from that dry run used fake risk tiers, so they are meaningless. **No setting was changed after this**; all monitor and twin settings had been saved to monitor_config.json and twin_config.json beforehand. The dry-run outputs were deleted. Future dry runs will use validation data relabelled as test.

- **05 Oct 2026, FINAL TEST RUN (one time, locked).** `final_evaluation.py` run by me on my computer; lock file `outputs/final_test/FINAL_RUN.lock` written. Nothing was tuned on these numbers. Full output in `outputs/final_test/final_results.txt`.
  - **Part 1: ECG on the test split (real data, n = 2,145, abnormal share 0.576)**, 95% bootstrap CIs:

  | Model | AUROC | AUPRC | Sensitivity at 90% specificity |
  |---|---|---|---|
  | Static only (age, sex) | 0.755 [0.734–0.775] | 0.804 | 0.436 |
  | **ECG-only ensemble (final model, D-12)** | **0.938 [0.928–0.947]** | **0.959 [0.952–0.966]** | **0.816 [0.789–0.852]** |
  | Fused (ECG + age + sex) | 0.940 [0.931–0.949] | 0.961 | 0.830 |

  - At the locked threshold 0.511 (chosen on validation): sensitivity 0.804, specificity 0.912. Confusion matrix: TN 829, FP 80, FN 242, TP 994.
  - Calibration: Brier 0.1011, ECE 0.0378, mean predicted 0.538 vs actual 0.576 (still mildly underconfident, L-7).
  - Single fold models: A 0.935, B 0.927, C 0.932, D 0.918; the 4-model average beats each one.
  - **Reading:** test AUROC 0.938 vs validation 0.942 is within run-to-run noise (about ±0.008), so validation did not flatter the model. Below the 0.97 leakage alarm. Specificity at the locked threshold (0.912) hit its 90% target on unseen patients. Fusion remains not clearly better (overlapping CIs), consistent with L-6 and D-12.
  - **Part 2: monitor on test (synthetic):** fixed baseline 0.44 false alarms/month, detection 65.1 / 84.5 / 96.6 / 99.6% (+3/+5/+8/+12 bpm); rolling baseline 0.51, detection 76.2 / 92.9 / 97.4 / 99.3%. Identical to the dry-run numbers disclosed above, as expected with fixed seeds.
  - **Part 3: twin on test:** 1,074 high-risk (50.1%), 1,071 low-risk. Mean detection: high risk uniform 92.4% vs risk-adjusted 93.9% (0.66 FA/month); low risk 90.6% vs 83.8% (0.31); all 91.4% vs 88.7% (0.49). Worst state: STABLE 486 (22.7%), WATCH 878 (40.9%), ESCALATE 781 (36.4%). Same pattern as validation (D-18).
  - **Model phase complete.** Next: doctor dashboard, then D-8 (repository), README, diagram, slides, video.

### D-19: Dashboard tool — Streamlit
- **Date:** 05 Oct 2026
- **Context:** the guidelines ask for a conceptual doctor dashboard. It must look convincing in the video and be runnable by a judge; the model phase is complete.
- **Options considered:**
  - **Option A: Streamlit (Python).** Reads our existing output files directly; pure Python; about 1 day. Cons: judges must install and run it (or watch the video); recognisable default look.
  - **Option B: static HTML page + exported JSON.** Opens with a double-click, can be hosted as a link. Cons: requires JavaScript (new skill), data not live from the model; about 1.5–2 days.
  - **Option C: Plotly Dash.** More layout control. Cons: more code (callbacks), slower to learn; about 1.5 days.
- **Decision:** Option A. Decided by me, on mentor recommendation.
- **Reasoning:** stays in Python, plugs into the files the pipeline already writes, fastest path, leaves time for README, slides and video.
- **Licence:** Apache License 2.0 (checked 05 Oct 2026 in the LICENSE file of github.com/streamlit/streamlit, first line). **Correction:** the mentor first said MIT when presenting the options; that was wrong.
- **Risks and limitations:** judges may not run it locally; mitigated by the video and, optionally, free hosting later (terms to be checked before relying on it).
- **Status:** Active

### D-20: Dashboard content — worklist + patient detail
- **Date:** 05 Oct 2026
- **Context:** decides what the doctor dashboard shows, how much there is to build, and what the video demonstrates.
- **Options considered:**
  - **Option A: single-patient view only** (~0.5 day). Smallest build, but no sense of "who needs me today".
  - **Option B: worklist + patient detail** (~1 day). Worklist sorted ESCALATE → WATCH → STABLE; detail page with both streams and a "why" panel showing which wearable measures moved. Matches clinical triage and answers the explainability question.
  - **Option C: B plus a "Model evidence" tab** (~1.25 days). Duplicates material that the slides and README will carry anyway.
- **Decision:** Option B. Decided by me, on mentor recommendation.
- **Reasoning:** shows the D-18 fusion in action and explains each alert; a fixed warning on every patient page carries the key limitations (L-8, L-9) instead of a whole tab. C's charts already exist as PNGs if wanted later.
- **Implementation details (mentor's choices within B, open to change):**
  - Test-split patients only (never seen during development). Nothing is re-tuned: all settings are read from `twin_config.json`, and the monitor functions are imported from `monitor.py`.
  - A "day" slider replays the 60 simulated days, so the video can show a patient moving STABLE → WATCH → ESCALATE.
  - A "Show simulation ground truth (demo only)" switch, off by default, reveals the injected episode and the true ECG label for the demo.
  - The "why" panel gives each measure's last-3-day average, the personal normal, the change in clinical units (bpm, h, %) and the shift in personal SDs.
  - The spread of the 4 fold models is shown as a simple uncertainty cue.
  - The 12-lead ECG is drawn from `X_test.npy` (memory-mapped, so it is not loaded into RAM).
- **Risks and limitations:**
  - Personal-SD values can look extreme (e.g. +20 SD), because simulated day-to-day wobble is small; the clinical-unit column is the one to read.
  - Sex labels use the inferred PTB-XL coding (0 = male), which is still to be confirmed.
- **Status:** Active

### D-21: Improvements after the competitor review
- **Date:** 06 Oct 2026
- **Context:** a review of 20 public competitor repositories (6 Oct 2026) showed that the strongest entries (a) check their wearable side on real data, (b) compare against a simple rule, (c) ship automated leakage tests, and (d) offer explanations. The PTB-XL test set is already opened, so no improvement may be judged on it; every change is checked on validation or on other data.
- **Options considered:**
  - **Option A: cheap items only** (simple-rule baseline, tests, dashboard polish). About 0.5 day. Leaves the "all wearable results are synthetic" gap open.
  - **Option B: A + run the locked monitor on real PMData + ECG saliency on the dashboard.** About 1.5 days; keeps a 1-day buffer.
  - **Option C: A + external mortality test on one part of CODE-15%.** About 2.5 days; the biggest possible upgrade but uses the whole buffer, with an uncertain result.
- **Decision:** Option B. Decided by me, on mentor recommendation.
- **Results:**
  - **Real-data check** (`real_data_check.py`, mentor's run on my processed PMData file). Rules fixed before running: locked settings, 3 measures (no HRV in PMData), gap rules stated in the script, every alarm counted as false. 15 of 16 people, 1,330 scored monitored days (44.3 person-months):

    | Setting | Alarms per person-month (real) | Budget on synthetic data | People with at least one alarm |
    |---|---|---|---|
    | Uniform (D-17) | 0.63 | 0.50 | 12 of 15 |
    | Low-risk tier (D-18) | 0.50 | 0.33 | 12 of 15 |
    | High-risk tier (D-18) | 0.90 | 0.66 | 13 of 15 |

    Reading: on real people the alarm rate is roughly 1.3–1.5× the synthetic budget. It is an upper bound (some alarms may be real illness or life events that cannot be verified), and PMData is not fully independent (the recipe was fitted on it). See L-10.
  - **Simple-rule baseline** (`baseline_rule.py`, validation only, synthetic). The two simple rules were fixed in advance; they are our illustrative choices, not a published guideline.

    | Method | False alarms / month | +3 | +5 | +8 | +12 bpm | Mean |
    |---|---|---|---|---|---|---|
    | RHR ≥ normal + 10 bpm, 3 days | 0.01 | 1.6% | 4.9% | 23.2% | 85.7% | 28.8% |
    | RHR ≥ normal + 5 bpm, 3 days | 0.10 | 29.0% | 65.2% | 93.8% | 98.9% | 71.7% |
    | Resting-HR-only monitor (tuned like D-17: threshold 1.60, k 3) | 0.49 | 74.1% | 86.8% | 98.1% | 99.2% | 89.5% |
    | Our monitor, 4 measures (D-17, locked) | 0.50 | 74.5% | 90.2% | 95.8% | 99.6% | 90.0% |

    Reading: the personal-baseline method catches far more small changes than fixed-bpm rules, at a higher but budgeted false-alarm rate. Adding HRV, sleep and steps to resting HR gives only a small gain in our simulation (L-11).
  - **Integrity tests** (`tests/test_integrity.py`, 9 tests, all passing on the mentor's copy):
    - no patient in two splits;
    - official folds and adults only;
    - the monitor never looks ahead;
    - no alarm before day 35;
    - episodes independent of the ECG label;
    - locked settings unchanged;
    - the risk threshold gives at least 90% specificity on validation;
    - the reported test AUROC and confusion matrix are recomputed from the saved per-patient predictions;
    - test set size and prevalence.
  - **ECG saliency** (`saliency.py`; dashboard shading toggle). Plain gradient of the 4-CNN ensemble, smoothed over 50 ms. Includes two checks:
    - a model-randomisation sanity check (a trained vs untrained network should give different maps);
    - a check that the recomputed probabilities equal the locked test predictions.
    **Results (my run, 06 Oct 2026):**
    - recomputed probabilities match the locked test predictions (max difference 2.97e-08);
    - model-randomisation check on 200 patients: median correlation between trained and untrained maps 0.11 (IQR 0.05 to 0.16), far below the 0.7 cut-off set in advance.
    So the maps reflect what the model learned, and the shading is kept as an optional toggle on the dashboard.
  - **Reproduced on my computer (06 Oct 2026):** real_data_check.py and baseline_rule.py printed exactly the numbers above, and all 9 integrity tests passed.
  - Dashboard: the ECG is now drawn full width.
- **Risks and limitations:** see L-10 and L-11. Saliency maps show sensitivity, not clinical reasoning, and are labelled that way on the dashboard.
- **Status:** Active

### D-22: Doctor-first dashboard redesign
- **Date:** 06 Oct 2026
- **Context:** I found the D-20 dashboard developer-oriented. It showed SD scores, thresholds, decision numbers, dataset IDs and default styling, and gave no plain-language summary and no way to act.
- **Options considered:**
  - **Option A: restyle only** (wording, hidden settings, theme). About 0.5 day; the layout would still feel like a data app.
  - **Option B: doctor-first redesign in Streamlit, same data and logic.** About 1 day.
  - **Option C: custom web app.** About 2–3 days; uses the whole buffer, adds a new language and repeats tested logic.
- **Decision:** Option B. Decided by me, on mentor recommendation.
- **What changed (dashboard.py, .streamlit/config.toml):**
  - Worklist: "Today's patients", with counts per status that carry a meaning line, status filters and a "hide reviewed" switch. Its columns are Status, Patient (pseudonymous PT-ID), Age, Sex, "Alert for N days", "What changed" in clinical units, Clinic ECG risk (High/Low with %), and Reviewed.
  - Patient page:
    - a header and a **one-sentence plain-language summary**, generated by a fixed template, not an LLM;
    - an ECG card (risk word, probability, agreement of the 4 model versions, and the 80% sensitivity caveat);
    - four reading tiles (last-3-day average vs usual);
    - one trend chart with a measure selector and the usual range;
    - the status strip and the 12-lead ECG, with the saliency option off by default;
    - "Mark as reviewed" with a note (session only).
  - SD scores, alert rules, thresholds and the source record ID were moved under "Technical details".
  - New "About this tool" page: how the status is decided, test performance read live from final_results.json, and limits.
  - Demo controls (day replay, simulation answers) are grouped and labelled as demo-only.
  - Calm clinical theme with a legible typeface (Atkinson Hyperlegible); colour is used only for status.
  - "What changed" lists shifts above 1 personal SD; while an alert is on, shifts above 0.5 SD are listed too, because the alert is driven by the average of the four measures (mentor's choice).
- **Unchanged:** all data, model outputs, monitor logic and locked settings. No treatment advice is shown (scope rule 7).
- **Status:** Active

### D-8: Repository layout (decided after the model was finished)
- **Date:** 06 Oct 2026 (deferred since 04 Oct)
- **Context:** the submission needs a public GitHub repository whose README covers team, college, title, problem statement, healthcare use case, technical stack, model details, a 15–20 minute unlisted video, licences, architecture diagram and presentation (PDF/PPT), all publicly accessible (Unstop submission form, seen 06 Oct 2026).
- **Options considered:**
  - **Option A: keep the scripts where they are**, adding `docs/`, `requirements.txt`, `.gitignore`, `LICENSE` and `README.md`. About 2 hours; nothing in the locked pipeline moves.
  - **Option B: restructure into `src/`.** About 0.5 day; every path changes and everything must be re-run.
  - **Option C: B plus a one-command runner.** About 1 day.
- **Decision:** Option A. Decided by me, on mentor recommendation.
- **Reasoning:** file locations stay exactly as they were when the locked test ran, so every number can be traced to its script.
- **Details:**
  - `outputs/` serves as the results folder; no duplicate `results/` copy, which could drift.
  - Raw data, PTB-XL `.npy` arrays, day-level PMData and per-person MMASH files are excluded by `.gitignore`. PMData files and per-person derived data are not redistributed (D-14a).
- **Note:** the submission form gives the video length as 15–20 minutes, while the earlier guideline text said at least 20; the video will target 19.5–20 minutes to satisfy both.
- **Status:** Active

### D-23: Code licence
- **Date:** 06 Oct 2026
- **Options considered:** MIT (short, permissive, most common); Apache 2.0 (adds an explicit patent grant; longer).
- **Decision:** MIT. Decided by me, on mentor recommendation.
- **Notes:**
  - Data keep their own licences (PTB-XL CC BY 4.0; PMData CC BY-NC 4.0 per its downloaded license.txt, while its web page states CC BY 4.0; MMASH ODbL v1.0).
  - Library licences checked by the mentor on the projects' GitHub LICENSE files on 06 Oct 2026: PyTorch BSD 3-Clause, wfdb MIT, Altair BSD 3-Clause, Streamlit Apache 2.0.
  - Citations checked on the dataset pages: the PhysioNet standard citation is now Pollard et al. 2026, Nature Health.
- **Status:** Active

### D-24: Learning twin — monitoring from day 0 instead of day 35
- **Date:** 14 Oct 2026 (design and pass/fail rules written BEFORE any learning-phase result was computed)
- **Context:** the locked monitor (D-17) needs 28 baseline days plus a 7-day gap, so it is blind for each patient's first 35 days. A patient discharged today would get no wearable monitoring for five weeks. The aim is a twin that starts from what is known about people like this patient and learns their own normal day by day.
- **Options considered:**
  - **Option A: shorter fixed window** (e.g. 14-day baseline). Simple, but still blind for 2–3 weeks, and a 14-day median/MAD is noisy.
  - **Option B: learning baseline (Bayesian shrinkage).** On day 0 the expected range comes from the population (recipe.json levels: resting HR by sex, HRV, sleep, steps). Each new day of the patient's own data pulls the estimate towards their personal level, and the expected range narrows. About 2 days.
  - **Option C: B plus an ECG-derived heart-rate anchor** (D-15 option C). Needs R-peak detection on PTB-XL and a link between clinic ECG heart rate and wearable resting HR that the simulator does not model. Deferred.
- **Decision:** Option B. Decided by me.
- **Method (fixed before running):**
  - Applies to days 0–34 only. **From day 35 the locked monitor (D-17, D-18) runs exactly as before**, so every locked and reported result stays valid. Runs of flagged days are counted within the learning phase only.
  - Same scales as monitor.py (HRV and steps on a log scale) and the same baseline window rule: the days from t−35 to t−8, but allowed to be shorter (0 to 28 days) while data accumulates.
  - Prior per measure from recipe.json: population mean and between-person SD of the personal level; population day-to-day SD and day-to-day memory (φ).
  - Personal level = precision-weighted average of the prior mean and the window mean, using the AR(1) effective sample size of the window (so correlated days count less).
  - Personal day-to-day SD = population value shrunk towards the window's SD, with the population value worth 10 days.
  - Daily z per measure = unhealthy direction × (value − personal level) / √(day-to-day variance + uncertainty of the level). Score = mean of the 4 z values, 3-day mean, exactly as monitor.py.
  - Alarm thresholds for the learning phase: tuned per risk tier on synthetic cohort A with the D-18 rule (highest mean detection within the tier's false-alarm budget: low risk 0.33, high risk 0.66 per patient-month; thresholds 0–4 in steps of 0.05; k = 1–6). Detection during tuning = first learning-phase flag on or after the episode start, before day 35.
- **Evaluation data (fixed before running):**
  - Two new synthetic cohorts built from the VALIDATION patients' age and sex with the unchanged simulator and recipe, but with episode start days 10–28: cohort A (seed 2401) for tuning, cohort B (seed 2402) for checking. The test set is not used.
  - Real PMData: each person's first 35 calendar days, 3 measures (no HRV), the D-21 gap rules (a day needs ≥ 2 measures; the 3-day mean needs ≥ 2 days; a missing day breaks a run). Every alarm counted as false.
- **Pass/fail rules (fixed before running):**
  - **R1 (false alarms, cohort B):** learning-phase false alarms ≤ 0.6 per patient-month overall (the 0.5 budget plus 20% for sampling noise between cohorts).
  - **R2 (benefit, cohort B):** mean detection within 14 days of episode start, averaged over the 4 sizes, at least 20 percentage points higher than the current system (locked monitor only, which cannot alarm before day 35).
  - **R3 (real people, PMData):** learning-phase alarm rate for each tier ≤ 2× that tier's synthetic budget (low ≤ 0.66, high ≤ 1.31 per person-month).
  - **Fallback:** if any rule fails, the learning phase is shown on the dashboard as an expected-range band only, with no alarms before day 35.
- **Risks and limitations:**
  - The prior comes from the same recipe that generated the synthetic patients, so the synthetic results are optimistic for the early days. The PMData check (R3) is the real-world test.
  - PMData people are Norwegian and mostly active, so their levels differ from the population prior; that makes R3 a harder test, not an easier one.
  - Test-split patients see the learning phase on the dashboard only for illustration; no test-set result is reported for it.
- **Results (my run, 14 Oct 2026; learning_twin.py, outputs/learning/learning_summary.txt):**
  - Tuned on cohort A: low-risk tier threshold 0.60, k = 3 (0.32 false alarms per patient-month); high-risk tier threshold 0.45, k = 3 (0.56).
  - **R1 PASS:** cohort B learning-phase false alarms 0.49 per patient-month (rule ≤ 0.60).
  - **R2 PASS:** episodes starting on days 10–28, caught within 14 days (cohort B, synthetic):

    | System | +3 bpm | +5 bpm | +8 bpm | +12 bpm | Mean |
    |---|---|---|---|---|---|
    | Current (locked monitor, blind before day 35) | 7.1% | 11.0% | 14.5% | 11.4% | 11.0% |
    | With the learning twin | 58.2% | 91.4% | 98.6% | 100.0% | 87.1% |

    Gain +76.1 points (rule ≥ +20). Median delay with the learning twin: 6, 5, 4 and 4 days.
  - **R3 PASS:** real PMData, first 35 days of 16 people (430 scored days = 14.3 person-months): low-risk tier 0.28 alarms per person-month (rule ≤ 0.66), high-risk tier 0.49 (rule ≤ 1.31).
  - All rules passed, so learning-phase alarms are switched on (`alarms_enabled: true` in learning_config.json).
  - Days 35–59 are unchanged: on the test-split dashboard data the flags for days 35–59 are identical with and without the learning twin, and the day-59 counts stay 449 Escalate, 980 Watch, 716 Stable.
  - 4 new integrity tests (no look-ahead, starts from the population prior, locked days untouched, settings locked); all 13 tests pass.
- **Reading:** the synthetic gain is large partly because the prior comes from the same recipe that generated the patients (see risks). The real-data check is the more meaningful result: on real people the learning phase did not raise more alarms than the budget allows.
- **Status:** Active

### D-25: Product-style doctor app (DilSe; first called CardioTwin, renamed because three other challenge repos use "CardioTwin", briefly Dhadkan, final name DilSe chosen 17 Oct)
- **Date:** 14 Oct 2026
- **Context:** the dashboard still read like a research tool (SD scores, thresholds and decision numbers on screen, raw ECG ids, no workflow). Note: the doctor-first features listed under D-22 (summary sentence, notes, "mark as reviewed", About page) were not present in `dashboard.py` on 14 Oct 2026; this entry implements them and more.
- **Options considered:**
  - **Option A: polish the Streamlit app** (language, workflow, layout, free hosting). About 3 days; reuses all tested logic.
  - **Option B: rebuild as a React web app.** About a week; slicker, but re-implements tested logic and risks breaking it.
- **Decision:** Option A. Decided by me.
- **What changed (`dashboard.py`, `requirements.txt`, `requirements-pipeline.txt`, `data/demo/ecg_demo.npz`):**
  - Clinical language: pseudonymous IDs (PT-00160), a one-line summary per patient, statistics moved under "Technical details".
  - Workflow: "new today" badges (status more urgent than yesterday), a suggested next step per status (Escalate: contact within 24 h; Watch: review in 3 days or check in within 3 days; Stable: routine; triage only, no treatment advice), mark as reviewed with a note, "hide reviewed", search, and a one-page PDF patient summary.
  - Twin profile card: ECG risk and model agreement, how much the twin has learned (D-24), the patient's alert setting in plain words, today's readings against usual.
  - Polish: header bar with clinic and date, status count cards, a 14-day heart-rate trend in every worklist row, charts that show data only up to "today", a status history with an event list, and a "How it works" page with test results and limits. Demo controls (day replay, simulation answers) grouped and labelled.
  - Hosting: the app now installs only numpy, pandas, matplotlib, streamlit and altair (`requirements.txt`); the pipeline packages moved to `requirements-pipeline.txt`. The real 12-lead ECGs and saliency maps of 99 demo test patients ship in `data/demo/ecg_demo.npz` (5 MB) so a hosted copy can draw them; the full `X_test.npy` is used automatically when present.
- **Unchanged:** all data, model outputs, monitor logic and locked settings. The status counts on every day are the same as before (day 59: 449 Escalate, 980 Watch, 716 Stable). Review notes are kept for the browser session only (no patient data is stored).
- **Risks and limitations:** the suggested next steps are fixed triage wording chosen by me, not clinical guidance; they must be agreed with clinicians before any real use.
- **Status names (17 Oct 2026):** the three daily states are shown as **Stable / Concern / Critical** (earlier shown as Stable / Watch / Escalate). Display names only: the code and saved outputs keep the internal codes STABLE / WATCH / ESCALATE, so no result changes.
- **Status:** Active

### D-26: ECG heart age, a supporting signal in the twin profile
- **Date:** 15 Oct 2026 (design and pass/fail rules written BEFORE the model was trained)
- **Context:** a competitor review (15 Oct) found entries whose wearable twin works much like ours (personal baseline, drift alarm, doctor dashboard). What no other public entry has is a model trained on real ECGs. Published work shows a CNN can estimate age from a 12-lead ECG, and that an ECG that looks older than the person is a risk signal: Attia et al. 2019 (Circ Arrhythm Electrophysiol, mean error about 7 years), Lima et al. 2021 (Nature Communications, ECG age more than 8 years above real age linked to higher mortality). Heart disease also starts earlier in India, so "how old does this heart look" is a meaningful, easy-to-explain addition to the twin.
- **Options considered:**
  - **Option A: no heart age.** Nothing to build, but the ECG stays a single probability.
  - **Option B: age CNN trained on all training ECGs.** More data, but abnormal hearts would partly be learned as "normal for their age", which shrinks the gap that carries the signal.
  - **Option C: age CNN trained on NORMAL training ECGs only.** The model learns what a healthy heart looks like at each age, so the gap reads as "older or younger than a healthy heart of the same age". Fewer training ECGs (7,156).
- **Decision:** Option C. Decided by me. It is a supporting signal: it does not change any alarm or status.
- **Method (fixed before training):**
  - Data: official folds 1–8, ECGs with label 0 (NORM), n = 7,156, for training. Validation NORM ECGs (n = 906) for early stopping and the bias correction. Test set scored exactly once at the end.
  - Signal: same preprocessing as the risk CNN (train-only per-lead normalisation, 100 Hz by averaging 5 samples, D-9).
  - Model: the plain CNN design of D-10 (widths 32/64/128/128, kernel 7), one output = age, trained on standardised age (training NORM mean and SD). L1 loss, Adam, learning rate 0.001, batch 64, seed 42, at most 30 epochs, early stopping with patience 5 on validation-NORM mean absolute error (MAE). One model, no ensemble, no augmentation.
  - Ages: PTB-XL codes ages over 89 as 300; prepare_data.py already stores them as 90. Kept as 90.
  - Bias correction (age models over-predict the young and under-predict the old): on validation NORM ECGs fit gap = c + d × age by least squares; corrected gap = (predicted − real age) − (c + d × age). **Heart age = real age + corrected gap**, rounded to whole years.
- **Pass/fail rules (fixed before training; test set, scored once):**
  - **R1 (accuracy):** MAE of the raw predicted age on test NORM ECGs ≤ 9.0 years, AND ≤ 0.70 × the MAE of always predicting the training-NORM mean age.
  - **R2 (meaning):** mean corrected gap of abnormal test ECGs minus that of normal test ECGs > 0, with the 95% bootstrap interval (2,000 resamples of patients, seed 7) entirely above 0.
  - **Fallback:** if R1 or R2 fails, heart age is NOT shown in the app and is reported in the README as a negative result.
- **Use in the twin (fixed before training):** shown in the patient profile and the PDF summary as context ("Heart age 61 · real age 52 · +9 years", with the typical error). It does not change thresholds, tiers or status, so every result in D-12 to D-24 stays valid.
- **Risks and limitations:**
  - PTB-XL is a German population recorded 1989–1996; the gap may be biased for Indian patients (L-3).
  - Ages over 89 are capped at 90, so the oldest patients look younger than they are.
  - The gap is not independent of the abnormal-ECG score: both read the same ECG. It is presented as an easier-to-understand view of the same heart, not as a second independent test.
  - One model on 7,156 ECGs; published models used hundreds of thousands, so a larger error is expected.
- **Results (run 15 Oct 2026, heart_age.py on a cloud CPU, outputs/heart_age/heart_age_summary.txt):**
  - Training stopped early at epoch 16; best epoch 11, validation NORM error 7.67 years. Bias correction (validation NORM): gap = 20.04 − 0.411 × age.
  - **R1 PASS:** test NORM ECGs (n = 909): mean error 7.85 years (rule ≤ 9.0), versus 13.40 for always guessing the mean age (ratio 0.59, rule ≤ 0.70); correlation 0.80. The model does read age from the ECG.
  - **R2 FAIL:** mean corrected gap abnormal −0.26 years, normal +0.53 years; difference −0.78 years (95% CI −1.49 to −0.09). Abnormal hearts did NOT look older than normal ones overall. 25% of both groups looked ≥ 5 years older.
  - **Fallback applied:** heart age is not shown in the app; reported in the README as a negative result. Nothing else in the system changes.
- **Exploratory reading (after the verdict; does not change it):** by age band, abnormal ECGs looked older than normal ones at 40–50 years (+4.4 vs +0.8) and 50–60 years (+4.0 vs +1.4), but younger above 80 (−5.2 vs −2.1). Abnormal patients are much older (mean 68 vs 52 years) and the training set has few healthy people over 75, with ages capped at 90, so the model cannot place very old hearts, and the oldest group dominates the abnormal average. A heart-age signal therefore looks plausible in middle age, but this would need a new, pre-registered test on data not yet used (for example an external ECG dataset), not a re-analysis of this test set.
- **Status:** Closed (negative result, kept for the record)
