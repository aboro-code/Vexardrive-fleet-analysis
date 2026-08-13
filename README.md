# VexarDrive Fleet Analysis

Driver Behaviour and Vehicle Health analytics for a two-wheeler fleet, built for the VexarDrive x Polaris Data Science Intern take-home assignment.

**Live Dashboard:** https://vexardrive-fleet-analysis-ep7y8trhzwgtob8znwsznp.streamlit.app/
**Technical Report:** https://docs.google.com/document/d/1ELjonmm3VqvoMYZohrXdb5jR8r824PFHK52h4lne7Eo/edit?tab=t.0

---

## Overview

One week of fleet telemetry — 30 drivers, 30 vehicles, 450 trips, ~13,000 telemetry readings — is used to build:

1. **Driver Behaviour Dashboard** — a composite risk score per driver, validated against independent K-means clustering (risk tiers).
2. **Vehicle Health Dashboard** — a composite wear/health score per vehicle, validated against Isolation Forest anomaly detection.

The approach follows a deliberate sequence: data quality validation → EDA → statistically-justified feature engineering → interpretable rule-based baseline → unsupervised ML layered on top to validate the baseline, rather than replacing interpretable logic with a black box. Every score and threshold is explicitly justified and documented — see the Technical Report and the notebooks' markdown cells for full reasoning.

## Key Findings

- Harsh-event thresholds (accel/gyro) were derived statistically (mean + 2σ) from the fleet-wide distribution, not picked arbitrarily.
- Driver roughness shows up as two distinct, only moderately correlated signal families — **magnitude** (how extreme) and **frequency** (how often) — so the risk score combines both.
- License experience has ~zero correlation with driving roughness in this dataset (tested, not assumed) and was excluded from scoring.
- Vehicle vibration correlates strongly with **days since last service** (Spearman 0.70), but not with vehicle age (-0.07) or weekly usage (-0.08) — service recency is the primary driver of the health score.
- K-means clustering (k=3, chosen via silhouette score) independently reproduced the same driver risk ordering as the rule-based score, with zero overlap between tiers.
- An initial Isolation Forest run on vehicle health data was uncorrelated with the baseline score (-0.066) and had to be corrected (fixed contamination + baseline-consistency constraint) before it produced a usable, explainable anomaly flag.

## Project Structure

```
vexardrive-fleet-analysis/
├── data/
│   └── VEXAR_Fleet_Dataset_CANDIDATE_VERSION.xlsx   # raw dataset (4 sheets)
├── notebooks/
│   ├── 01_eda.ipynb                  # data quality checks, distributions, driver/vehicle signal validation
│   ├── 02_feature_engineering.ipynb  # harsh-event thresholds, features, baseline risk & health scores
│   └── 03_modeling_dashboards.ipynb  # K-means driver clustering, Isolation Forest anomaly detection
├── app/
│   └── app.py                        # Streamlit dashboard (Driver Behaviour + Vehicle Health)
├── src/
│   ├── data_loader.py                # shared raw-data loading logic (header=2 for all sheets)
│   └── __init__.py
├── outputs/
│   ├── driver_features_final.csv     # final driver feature table incl. risk_score, risk_tier
│   └── vehicle_health_features_final.csv  # final vehicle feature table incl. health_score, anomaly_flag
├── requirements.txt
└── README.md
```

## Methodology Summary

| Stage | What was done |
|---|---|
| Data quality | Validated nulls, duplicate IDs, foreign key consistency, and logical constraints across all 4 sheets — no issues found. |
| EDA | Distribution analysis of trips and telemetry; identified the tight-baseline / sparse-tail shape in accel & gyro signals that justified statistical harsh-event thresholds. |
| Feature engineering | Harsh-event thresholds (mean + 2σ); per-trip and per-driver harsh-event rates; per-vehicle vibration signatures. |
| Baseline scoring | Driver risk score = weighted average (25% each) of 4 normalized features. Vehicle health score = 60% vibration + 40% days-since-service. |
| ML validation | K-means (k=3, silhouette-selected) for driver risk tiers; Isolation Forest (corrected) for vehicle anomaly flags — both validated against the interpretable baseline rather than trusted blindly. |
| Dashboards | Streamlit, two pages, deployed to Streamlit Community Cloud. |

Full reasoning, including two documented cases where an initial hypothesis or model output didn't hold up and was corrected after validation, is in the notebooks' markdown cells and the Technical Report.

## Setup & Reproduction

```bash
# clone and enter the repo
git clone https://github.com/aboro-code/Vexardrive-fleet-analysis.git
cd Vexardrive-fleet-analysis

# create and activate a virtual environment
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
# or venv\Scripts\Activate.ps1   # Windows (PowerShell)

# install dependencies
pip install -r requirements.txt

# run the notebooks in order (01 -> 02 -> 03), or just run the dashboard directly:
streamlit run app/app.py
```

The dashboard reads `outputs/driver_features_final.csv` and `outputs/vehicle_health_features_final.csv` directly, so it can be run without re-executing the notebooks.

## Tech Stack

Python, pandas, numpy, matplotlib, seaborn, scikit-learn (KMeans, IsolationForest), Streamlit, Plotly, Jupyter.

## Other Potential Uses of This Dataset

- **Predictive maintenance** — the vibration ↔ days-since-service relationship suggests a time-series model could predict service needs ahead of a fixed schedule.
- **Driver coaching & gamification** — risk tiers and harsh-event trends could power driver-facing feedback and incentive programs.
- **Usage-based insurance pricing** — individual risk scores could inform premium calculations.
- **Trip-log fraud / integrity detection** — the same anomaly-detection approach could flag implausible trip/GPS patterns.
- **Route optimization** — pairing driver risk profiles with route data to assign safer drivers to higher-risk routes.

## Author

Arnab Boro — [github.com/aboro-code](https://github.com/aboro-code)
