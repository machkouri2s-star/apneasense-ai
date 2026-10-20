# ApneaSense AI

Explainable sleep apnea detection from overnight ECG.
AI Innovation Hackathon 2026, Biomedical AI, Challenge 4.
Research and educational prototype. Not a diagnostic tool.

## What it does
Reads an overnight ECG recording (PhysioNet Apnea-ECG), predicts apnea minute by minute,
merges predictions into events (start, end, duration) and shows them on an interactive
timeline with model confidence and low-confidence flags.

## Method
1. R-peak annotations give RR intervals (time between heartbeats).
2. 60 s windows aligned with expert annotations.
3. Features per minute: heart-rate variability (mean RR, SDNN, RMSSD, pNN50...), frequency-band
   power (VLF, LF, HF) over a 5-minute context, per-recording normalization (z-scores),
   and neighboring-minute context.
4. Model: XGBoost with class weighting.
5. Post-processing: consecutive positive minutes become events; minimum event length is adjustable.

## Validation
5-fold cross-validation grouped by recording: no night appears in both training and test.
Data: 35 learning-set recordings of the Apnea-ECG database.

Minute-level results on unseen nights:
- Accuracy 0.788
- Sensitivity 0.680
- Specificity 0.854
- F1 0.711
- AUROC 0.859

Event-level metrics and a per-recording view are in the app (Scientific validation tab).

## Limitations
- Single ECG-derived signal; no respiration or SpO2.
- Sensitivity of 0.68 means about one third of apnea minutes are missed.
- Minute labels come from one database; cross-database robustness is not tested.
- Not calibrated for clinical use. No diagnostic claims.

## Run it
    pip install -r requirements.txt
    streamlit run app.py

To rebuild everything from raw data:
    python download3.py
    python features.py
    python train.py

## Files
- download3.py: downloads annotations (.apn, .qrs)
- features.py: builds per-minute features (features.csv)
- train.py: cross-validated training, writes predictions.csv and model.json
- app.py: Streamlit dashboard
