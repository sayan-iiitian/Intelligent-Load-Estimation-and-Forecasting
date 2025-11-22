import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import tempfile
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

st.set_page_config(page_title="Load Forecast Demo", layout="wide")

# Utilities
def compute_metrics(y_true, y_pred):
    mask = (np.array(y_true) != 0)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((np.array(y_true)[mask] - np.array(y_pred)[mask]) / np.array(y_true)[mask]))*100 if mask.any() else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape}

def plot_series(idx_true, y_true, idx_pred, y_pred, title="Forecast vs Actual"):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(idx_true, y_true, label="Actual", linewidth=1.5)
    ax.plot(idx_pred, y_pred, label="Forecast", linewidth=1.5)
    ax.legend(); ax.set_title(title)
    return fig

def plot_residuals(residuals):
    fig, axes = plt.subplots(1,3, figsize=(15,3))
    axes[0].plot(residuals); axes[0].set_title("Residuals")
    try:
        import statsmodels.api as sm
        sm.graphics.tsa.plot_acf(residuals.dropna(), lags=48, ax=axes[1])
        sm.graphics.tsa.plot_pacf(residuals.dropna(), lags=48, ax=axes[2])
    except Exception:
        axes[1].text(0.2, 0.5, "ACF unavailable", fontsize=12); axes[2].text(0.2, 0.5, "PACF unavailable", fontsize=12)
    plt.tight_layout()
    return fig

# Data handling
def read_csv_to_df(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, parse_dates=True, infer_datetime_format=True)
        # If there's a first column that looks like datetime, set it as index
        if df.shape[1] > 0 and df.columns[0].lower() in ["timestamp","date","datetime","time","index"]:
            df = df.set_index(df.columns[0])
        # If no datetime index, try to parse the first col as date
        if not isinstance(df.index, pd.DatetimeIndex):
            # try to find a datetime column
            for c in df.columns:
                try:
                    parsed = pd.to_datetime(df[c])
                    df = df.set_index(parsed)
                    break
                except Exception:
                    continue
        df = df.sort_index()
        return df
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

def prepare_series(df):
    if 'load' not in df.columns and df.shape[1] >= 1:
        df = df.rename(columns={df.columns[0]:'load'})
    df = df.asfreq(pd.infer_freq(df.index) or 'H')
    return df

# Modeling functions
def fit_arima_and_predict(series, exog=None, test_h=24*7, seasonal_period=24, auto_args=None):
    import pmdarima as pm
    train = series.iloc[:-test_h]
    test = series.iloc[-test_h:]
    if exog is not None:
        train_exog = exog.iloc[:-test_h]
        test_exog = exog.iloc[-test_h:]
    else:
        train_exog = None; test_exog = None
    auto_args = auto_args or {}
    model = pm.auto_arima(train, exogenous=train_exog, seasonal=True, m=seasonal_period, stepwise=True, suppress_warnings=True, error_action='ignore', **auto_args)
    forecast = model.predict(n_periods=len(test), exogenous=test_exog)
    return model, test.index, test.values, forecast

def fit_neuralprophet_and_predict(df, test_h, epochs=50):
    from neuralprophet import NeuralProphet
    import torch.serialization
    import neuralprophet.configure
    # Fix for PyTorch 2.6+ weights_only=True default
    try:
        torch.serialization.add_safe_globals([
            neuralprophet.configure.ConfigSeasonality,
            neuralprophet.configure.Season
        ])
    except Exception:
        pass
    train_df = df.reset_index().rename(columns={'index':'ds'})
    if 'load' in train_df.columns:
        train_df = train_df.rename(columns={'load':'y'})
    for reg in ['temp','solar','wind']:
        if reg not in train_df.columns:
            train_df[reg] = 0.0
    m = NeuralProphet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, n_lags=0, n_forecasts=1)
    m = m.add_future_regressor('temp'); m = m.add_future_regressor('solar'); m = m.add_future_regressor('wind')
    m.fit(train_df[:-test_h], freq='H', epochs=epochs)
    future = train_df[-test_h:][['ds','temp','solar','wind']]
    forecast = m.predict(future)
    if 'yhat1' in forecast.columns:
        pred = forecast['yhat1'].values
    elif 'yhat' in forecast.columns:
        pred = forecast['yhat'].values
    else:
        pred = forecast.iloc[:, forecast.columns.str.startswith('yhat')].iloc[:,0].values
    return m, pd.to_datetime(future['ds']).values, df['load'].values[-test_h:], pred

def fit_lstm_and_predict(df, seq_len=168, epochs=20, batch_size=64):
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    features = ['load','temp','solar','wind','hour','dayofweek']
    df2 = df.copy()
    df2['hour'] = df2.index.hour
    df2['dayofweek'] = df2.index.dayofweek
    for c in ['temp','solar','wind']:
        if c not in df2.columns:
            df2[c] = 0.0
    arr = df2[features].values
    train_len = int(0.85 * len(arr))
    mean = arr[:train_len].mean(axis=0); std = arr[:train_len].std(axis=0)+1e-6
    arr_norm = (arr - mean)/std
    X=[]; Y=[]
    for i in range(seq_len, len(arr_norm)):
        X.append(arr_norm[i-seq_len:i])
        Y.append(arr_norm[i,0])
    X = np.array(X); Y = np.array(Y)
    split_i = train_len - seq_len
    if split_i <= 0:
        raise RuntimeError("Not enough data for LSTM sequences.")
    X_train, X_test = X[:split_i], X[split_i:]
    Y_train, Y_test = Y[:split_i], Y[split_i:]
    model = models.Sequential([layers.Input(shape=(seq_len, X.shape[2])), layers.LSTM(64), layers.Dense(32,activation='relu'), layers.Dense(1)])
    def rmse(y_true,y_pred): return tf.sqrt(tf.reduce_mean(tf.square(y_pred-y_true)))
    def r2(y_true,y_pred):
        ss_res = tf.reduce_sum(tf.square(y_true-y_pred))
        ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        return 1 - ss_res/(ss_tot+1e-6)
    model.compile(optimizer='adam', loss='mse', metrics=['mae', rmse, r2])
    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
    pred_norm = model.predict(X_test).flatten()
    pred = pred_norm * std[0] + mean[0]
    idx_test = df.index[-len(pred):]
    return model, history, idx_test, df['load'].values[-len(pred):], pred

def fit_hybrid_and_predict(df, base_pred, seq_len=168, epochs=10):
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    features_res = ['temp','solar','wind','hour','dayofweek']
    df2 = df.copy()
    df2['hour'] = df2.index.hour; df2['dayofweek'] = df2.index.dayofweek
    for c in ['temp','solar','wind']:
        if c not in df2.columns:
            df2[c] = 0.0
    train = df2
    train_resid = train['load'] - train['load'].rolling(24, min_periods=1).mean()
    arr_res = train[features_res].values
    y_res = train_resid.values
    mean_res = arr_res.mean(axis=0); std_res = arr_res.std(axis=0)+1e-6
    arr_res_norm = (arr_res - mean_res)/std_res
    Xr=[]; Yr=[]
    for i in range(seq_len, len(arr_res_norm)):
        Xr.append(arr_res_norm[i-seq_len:i]); Yr.append(y_res[i])
    Xr = np.array(Xr); Yr = np.array(Yr)
    if len(Xr) <= 20:
        raise RuntimeError("Not enough data for hybrid LSTM training.")
    def rmse(y_true,y_pred): return tf.sqrt(tf.reduce_mean(tf.square(y_pred-y_true)))
    def r2(y_true,y_pred):
        ss_res = tf.reduce_sum(tf.square(y_true-y_pred))
        ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        return 1 - ss_res/(ss_tot+1e-6)
    model_r = models.Sequential([layers.Input(shape=(seq_len, Xr.shape[2])), layers.LSTM(32), layers.Dense(16, activation='relu'), layers.Dense(1)])
    model_r.compile(optimizer='adam', loss='mse', metrics=['mae', rmse, r2])
    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    hist = model_r.fit(Xr, Yr, validation_split=0.1, epochs=epochs, batch_size=64, callbacks=[es], verbose=0)
    combined_features = pd.concat([train[features_res].iloc[-(seq_len+len(base_pred)):], train[features_res].iloc[-len(base_pred):]])
    comb_arr = (combined_features.values - mean_res)/std_res
    X_comb=[]
    for i in range(seq_len, seq_len + len(base_pred)):
        X_comb.append(comb_arr[i-seq_len:i])
    X_comb = np.array(X_comb)
    pred_resid = model_r.predict(X_comb).flatten()
    hybrid_pred = np.array(base_pred) + pred_resid[:len(base_pred)]
    idx_test = df.index[-len(hybrid_pred):]
    return model_r, hist, idx_test, df['load'].values[-len(hybrid_pred):], hybrid_pred

# Streamlit UI
st.title("Smart Grid Load Forecast — Demo")
st.markdown("Upload dataset with a datetime column and `load` column. Optional regressors: `temp`, `solar`, `wind`.")

uploaded = st.file_uploader("Upload CSV (or leave blank to use synthetic sample)", type=["csv"])
if uploaded is not None:
    df = read_csv_to_df(uploaded)
    if df is None:
        st.stop()
else:
    st.info("No file uploaded — using synthetic sample data (hourly ~180 days).")
    rng = pd.date_range("2024-01-01", periods=24*180, freq='H')
    base = 100 + 10*np.sin(2*np.pi * rng.hour / 24)
    seasonal = 20*np.sin(2*np.pi * rng.dayofyear / 365)
    noise = np.random.normal(0,3,len(rng))
    solar = np.clip(30*np.sin((rng.hour-6)/24 * 2*np.pi), 0, None) + np.random.normal(0,2,len(rng))
    wind = np.abs(np.random.normal(5,2,len(rng)))
    df = pd.DataFrame({'load': base + seasonal + 0.2*solar - 0.1*wind + noise, 'temp': 20 + np.random.normal(0,1,len(rng)), 'solar': solar, 'wind': wind}, index=rng)

df = prepare_series(df)
st.write("Data sample:")
st.dataframe(df.head(6))

model_choice = st.selectbox("Model", ["ARIMA", "Prophet", "LSTM", "Hybrid (base NP/ARIMA + LSTM residual)"])
h_days = st.number_input("Forecast horizon (days)", min_value=1, max_value=30, value=1)
test_h = int(h_days * 24)

run_button = st.button("Run Forecast")

if run_button:
    with st.spinner("Running model... this may take time depending on choice"):
        try:
            if model_choice == "ARIMA":
                exog = df[['temp','solar','wind']].copy() if set(['temp','solar','wind']).issubset(df.columns) else None
                model, idx_test, y_test, pred = fit_arima_and_predict(df['load'], exog=exog, test_h=test_h)
                metrics = compute_metrics(y_test, pred)
                st.write("Metrics:", metrics)
                st.pyplot(plot_series(idx_test, y_test, idx_test, pred, title="ARIMA Forecast"))
                st.pyplot(plot_residuals(pd.Series(y_test - pred)))
                out = pd.DataFrame({'actual': y_test, 'forecast_arima': pred}, index=idx_test)
            elif model_choice == "Prophet":
                m, idx_test, y_test, pred = fit_neuralprophet_and_predict(df, test_h=test_h, epochs=50)
                metrics = compute_metrics(y_test, pred)
                st.write("Metrics:", metrics)
                st.pyplot(plot_series(idx_test, y_test, idx_test, pred, title="Prophet Forecast"))
                out = pd.DataFrame({'actual': y_test, 'forecast_np': pred}, index=idx_test)
            elif model_choice == "LSTM":
                model, history, idx_test, y_test, pred = fit_lstm_and_predict(df, seq_len=168, epochs=30, batch_size=64)
                metrics = compute_metrics(y_test, pred)
                st.write("Metrics:", metrics)
                st.pyplot(plot_series(idx_test, y_test, idx_test, pred, title="LSTM Forecast"))
                st.pyplot(plot_residuals(pd.Series(y_test - pred)))
                st.write("Training metrics (last epoch):", {k: history.history[k][-1] for k in history.history})
                out = pd.DataFrame({'actual': y_test, 'forecast_lstm': pred}, index=idx_test)
            elif model_choice == "Hybrid (base NP/ARIMA + LSTM residual)":
                # choose base: try Prophet then ARIMA
                base_name = st.radio("Hybrid base model", options=["Prophet","ARIMA"], index=0)
                if base_name == "Prophet":
                    try:
                        m, idx_b, y_b, base_pred = fit_neuralprophet_and_predict(df, test_h=test_h, epochs=40)
                    except Exception as e:
                        st.error(f"Prophet base failed: {e}")
                        base_pred = None
                else:
                    try:
                        exog = df[['temp','solar','wind']].copy() if set(['temp','solar','wind']).issubset(df.columns) else None
                        model_arima, idx_b, y_b, base_pred = fit_arima_and_predict(df['load'], exog=exog, test_h=test_h)
                    except Exception as e:
                        st.error(f"ARIMA base failed: {e}")
                        base_pred = None
                if base_pred is None:
                    st.stop()
                model_r, hist, idx_test, y_test, hybrid_pred = fit_hybrid_and_predict(df, base_pred, seq_len=168, epochs=20)
                metrics = compute_metrics(y_test, hybrid_pred)
                st.write("Metrics:", metrics)
                st.pyplot(plot_series(idx_test, y_test, idx_test, hybrid_pred, title=f"Hybrid Forecast (base={base_name})"))
                st.pyplot(plot_residuals(pd.Series(y_test - hybrid_pred)))
                st.write("Training metrics (last epoch):", {k: hist.history[k][-1] for k in hist.history})
                out = pd.DataFrame({'actual': y_test, 'forecast_hybrid': hybrid_pred}, index=idx_test)

            # provide download button
            csv = out.to_csv().encode()
            st.download_button("Download forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Model run failed: {type(e).__name__} {e}")