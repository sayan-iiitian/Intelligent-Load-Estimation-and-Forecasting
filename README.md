# Spatio-Temporal Load Estimation & Forecasting for Smart Grids

This repository contains the research artifacts, notebooks, and Streamlit demo for **“Spatio-Temporal Load Estimation and Short-Term Forecasting using AI Models for Smart Grids.”** The project was executed by Group 3 (Avighyan Chakraborty, Rihan Ghosh, Shubhankar Kumar, Sayan Mandal) at **IIIT Kalyani** under the supervision of **Dr. Debashish Bera** as part of the Data Analytics and Optimization course.

The goal is to deliver a resilient forecasting pipeline that can **estimate missing load data** and produce **1-day to 7-day ahead forecasts** for grids facing high renewable penetration, seasonal swings, and data delays.

---

## Highlights
- **AI Model Benchmarks:** ARIMA, Prophet (NeuralProphet variant), LSTM, and an experimental hybrid (Prophet/ARIMA + residual LSTM).
- **End-to-end pipeline:** Missing-value imputation, feature engineering, model training, evaluation, and visualization in one workspace.
- **Smart-grid context:** Data inspired by Amendis (Tetouan, Morocco) SCADA feeds enriched with weather, renewable, and socioeconomic cues.
- **Interactive UI:** `app.py` hosts a Streamlit app for uploading new datasets, running models, visualizing forecasts, and exporting results.
- **Deployed demo:** https://daaproject-xmmyc8dwniphygwwdwtp88.streamlit.app/

---

## Repository Structure

| Path | Purpose |
| --- | --- |
| `Spatio_Temporal_Load_Forecasting.ipynb` | Research notebook covering preprocessing, Prophet/ARIMA/LSTM training, plots, and metric dumps. |
| `app.py` | Streamlit interface that wraps ARIMA, NeuralProphet, LSTM, and hybrid residual models for quick experimentation. |
| `synthetic_load_with_missing.csv` | Sample hourly dataset (≈180 days) with intentional gaps for imputation & forecasting demos. |
| `Report.docx` | Formal project report (executive summary, objectives, method, results, references). |
| `Presentation.pdf` | Slide deck summarizing motivation, data, methods, metrics, and conclusions. |
| `requirements.txt` | Python dependencies for the notebook and Streamlit application. |

---

## Data & Preprocessing
- **Source context:** Tetouan (northern Morocco) smart grid managed by Amendis; three primary substations (Quads, Smir, Boussafou). Population ≈550k with Mediterranean climate.
- **Core features:** `timestamp`, `load (kW/MW)`, `temperature (°C)`, `solar_radiation (W/m²)`, `wind_speed (m/s)` plus derived calendar fields (`hour`, `dayofweek`).
- **Missing values:** Filled through **ARIMA(2,1,2)** predictions to preserve temporal continuity before modeling.
- **Frequency alignment:** Hourly cadence enforced via `pandas.asfreq`. Optional regressors (temp/solar/wind) are zero-filled when absent to keep neural models stable.
- **Scenario coverage:** Seasonal variation, demand spikes (events/holidays), renewable fluctuations—all used to stress-test robustness.

---

## Modeling Stack

| Model | Why it’s here | Notes |
| --- | --- | --- |
| **ARIMA / auto_arima** | Baseline statistical model for stationarized series. | Captures trend/seasonality, supports exogenous weather regressors, good explainability. |
| **Prophet / NeuralProphet** | Handles multiple seasonalities, missing data, changepoints. | Implemented via NeuralProphet for easier deployment and regressor support. |
| **LSTM** | Learns non-linear, long-range patterns inherent in smart-grid load. | Uses sliding windows (seq_len=168), normalized features, early stopping. |
| **Hybrid (Base + Residual LSTM)** | Experimental stack to correct systematic residuals. | Base forecast from Prophet or ARIMA, residual modeled via LSTM on meteorological features. |

Supporting tooling includes `statsmodels`, `pmdarima`, `tensorflow/keras`, `neuralprophet`, `pytorch-lightning`, `scikit-learn`, `matplotlib`, `seaborn`, and `streamlit`.

---

## Performance Snapshot

Evaluation uses **MAE**, **MAPE**, and **RMSE** on held-out horizons (1-day and 7-day). The following metrics come from the baseline benchmark reported in both the notebook and documentation:

| Model | MAE | MAPE (%) | RMSE |
| --- | --- | --- | --- |
| Naive (prev-day) | 4.435 | 7.473 | 15.417 |
| ARIMA | 4.088 | 6.388 | 13.472 |
| Prophet | 2.756 | 4.986 | 9.285 |
| **LSTM** | **2.063** | **3.025** | **7.648** |
| Hybrid (Prophet + LSTM residual) | 12.188 | 13.928 | 46.907 |

Key takeaways:
- **LSTM dominates** across all metrics, confirming its ability to capture complex spatial-temporal cues.
- **Prophet** remains highly competitive and far more interpretable, making it attractive for operators who need seasonal decomposition.
- **ARIMA** improves on naive baselines but lags behind AI models when the series exhibits strong non-linearities.
- **Hybrid** pipeline needs more tuning—the current run surfaced issues when chaining Prophet outputs into residual LSTMs.

Additional diagnostics (e.g., confusion-matrix-style correctness tables in `Presentation.pdf`) reinforce that LSTM and Prophet track peaks/troughs more faithfully than other approaches.

---

## Running the Streamlit Demo

1. **Set up environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```
2. **Launch the app**
   ```bash
   streamlit run app.py
   ```
3. **Use the UI**
   - Upload a CSV with a datetime column (auto-detected) and a `load` column. Optional columns: `temp`, `solar`, `wind`.
   - Select model (`ARIMA`, `Prophet`, `LSTM`, or `Hybrid`), choose forecast horizon (days), and hit **Run Forecast**.
   - Inspect metrics, residual plots, and download the forecast CSV for downstream analysis.

> Tip: If you skip the upload, the app falls back to `synthetic_load_with_missing.csv`, which simulates 6 months of hourly data with seasonal, solar, and wind effects.

---

## Reproducing Notebook Experiments

1. Open `Spatio_Temporal_Load_Forecasting.ipynb` in Jupyter or VS Code.
2. Install notebook dependencies (same as `requirements.txt`, plus `pykalman`, `Prophet`, `pmdarima` if missing).
3. Run cells sequentially:
   - Load `synthetic_load_with_missing.csv`.
   - Perform ARIMA-based imputation.
   - Train Prophet/ARIMA/LSTM models.
   - Visualize predictions and export metrics.

Notebook markdown cells document each stage (e.g., “Missing Value Preprocessing,” “Prophet Estimation,” “ARIMA Estimation,” “LSTM Estimation”) for quick navigation.

---

## Documentation & Presentation

- **Report:** `Report.docx` contains the executive summary, objectives, dataset description, preprocessing visuals, model deep dives, quantitative results, deployment link, and references.
- **Presentation:** `Presentation.pdf` mirrors the report for stakeholder briefings—project overview, objectives, data description, preprocessing snapshots, metric tables, confusion-matrix view, and conclusions highlighting ALDC (Automatic Load Dispatch Center) impact.

Both files are the primary sources for narratives included in this README.

---

## Future Enhancements
- Stabilize the hybrid Prophet/ARIMA + LSTM residual architecture and re-benchmark.
- Incorporate additional spatial signals (feeder-level anomalies, GIS layers) to extend beyond single-city data.
- Automate hyper-parameter tuning (Optuna/Ray Tune) for LSTM and Prophet seasonality settings.
- Containerize the Streamlit app and schedule periodic retraining with fresh SCADA feeds.

---

## References
- https://www.kaggle.com/datasets/fedesoriano/electric-power-consumption  
- Christopher Olah, *Understanding LSTMs*, https://colah.github.io/posts/2015-08-Understanding-LSTMs/  
- https://www.sciencedirect.com/science/article/pii/S2352484724004566  
- https://www.sciencedirect.com/science/article/abs/pii/S095965262301288X

---

For questions or collaboration requests, feel free to reach out to any Group 3 member. Happy forecasting!

