# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="STQA3013 Interactive Backtesting & Portfolio Dashboard",
    layout="wide",
)

INITIAL_CAPITAL = 10000
RISK_FREE_RATE = 0.03
TRANSACTION_COST = 0.002

files = {
    "FBMKLCI": "FBMKLCI.csv",
    "GOLD": "GOLD.csv",
    "BITCOIN": "BITCOIN.csv",
    "ABF": "ABF.csv",
    "MAYBANK": "MAYBANK.csv",
    "TENAGA": "TENAGA.csv",
    "FCPO": "FCPO.csv",
}


# ==========================================================
# DATA LOADING & CLEANING
# ==========================================================
@st.cache_data
def load_datasets():
  loaded = {}
  for asset, filename in files.items():
    if not os.path.exists(filename):
      continue
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = (
        df["Price"].astype(str).str.replace(",", "", regex=False).str.strip()
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.sort_values("Date").dropna(subset=["Date", "Price"])
    loaded[asset] = df
  return loaded


def add_indicators(df):
  df = df.copy()
  df["MA50"] = df["Price"].rolling(50).mean()
  df["MA200"] = df["Price"].rolling(200).mean()
  ema12 = df["Price"].ewm(span=12, adjust=False).mean()
  ema26 = df["Price"].ewm(span=26, adjust=False).mean()
  df["MACD"] = ema12 - ema26
  delta = df["Price"].diff()
  gain = delta.clip(lower=0)
  loss = -delta.clip(upper=0)
  rs = gain.rolling(14).mean() / loss.rolling(14).mean()
  df["RSI"] = 100 - 100 / (1 + rs)
  return df


def human_strategy(df):
  df = add_indicators(df)
  df["Signal"] = np.where(df["MA50"] > df["MA200"], 1, 0)
  return df


def ai_strategy(df):
  df = add_indicators(df)
  df["Signal"] = np.where((df["RSI"] < 35) & (df["MACD"] > 0), 1, 0)
  return df


def benchmark_strategy(df):
  df = df.copy()
  df["Signal"] = 1
  return df


def split_data(df):
  df = df.copy()
  df["Date"] = pd.to_datetime(df["Date"])
  df = df.sort_values("Date")
  df.set_index("Date", inplace=True)
  train = df.loc["2016-01-01":"2024-12-31"]
  validation = df.loc["2025-01-01":"2025-12-31"]
  forward = df.loc["2026-01-01":"2026-09-30"]
  return train, validation, forward, df


def run_backtest(df):
  df = df.copy()
  df["Return"] = df["Price"].pct_change()
  # Next-day execution to avoid look-ahead bias
  df["Position"] = df["Signal"].shift(1).fillna(0)
  trades = df["Position"].diff().abs().fillna(0)
  cost = trades * TRANSACTION_COST
  df["Strategy_Return"] = df["Position"] * df["Return"] - cost
  df["Portfolio"] = (
      INITIAL_CAPITAL * (1 + df["Strategy_Return"].fillna(0)).cumprod()
  )
  return df


def performance_metrics(df, is_benchmark=False):
  portfolio = df["Portfolio"].dropna()
  if len(portfolio) < 2:
    return {
        "Return %": np.nan,
        "Annual Return %": np.nan,
        "Sharpe": np.nan,
        "Max DD %": np.nan,
        "Trades": 0,
        "Win Rate %": np.nan,
    }

  total_return = portfolio.iloc[-1] / portfolio.iloc[0] - 1
  annual_return = (1 + total_return) ** (252 / len(portfolio)) - 1
  daily = portfolio.pct_change().dropna()
  annual_volatility = daily.std() * np.sqrt(252)

  if np.isnan(annual_volatility) or annual_volatility < 0.05:
    sharpe = 0.0
  else:
    sharpe = (daily.mean() * 252 - RISK_FREE_RATE) / annual_volatility
  sharpe = max(min(sharpe, 5.0), -5.0)

  rolling_peak = portfolio.cummax()
  drawdown = (portfolio - rolling_peak) / rolling_peak
  max_drawdown = drawdown.min()

  if is_benchmark:
    num_trades = 1
    win_rate_val = np.nan
  else:
    entries = (
        (df["Position"] == 1) & (df["Position"].shift(1).fillna(0) == 0)
    ).sum()
    num_trades = int(entries)
    wins = (daily > 0).sum()
    losses = (daily < 0).sum()
    active_days = wins + losses
    win_rate_val = (
        round((wins / active_days) * 100, 2) if active_days > 0 else 0.0
    )

  return {
      "Return %": round(total_return * 100, 2),
      "Annual Return %": round(annual_return * 100, 2),
      "Sharpe": round(sharpe, 2),
      "Max DD %": round(max_drawdown * 100, 2),
      "Trades": num_trades,
      "Win Rate %": win_rate_val,
  }


def build_portfolio_schemes(
    datasets, selected_assets, period_name, best_strategy_lookup
):
  """
    Combine best strategy per asset into one portfolio and compare 3 weighting schemes:
      1. Equal Weight (1/N)
      2. Inverse-Volatility Weight (1/sigma_i normalized; estimated from Training/Validation volatility to avoid look-ahead)
      3. AI-Suggested Risk-Adjusted Momentum Weight (Sharpe-tilted inverse-volatility weighting from Validation/Training)
    """
  strategy_map = {
      "Human": human_strategy,
      "AI": ai_strategy,
      "Benchmark": benchmark_strategy,
  }

  strat_returns = {}
  inv_vol_raw = {}
  ai_score_raw = {}

  for asset in selected_assets:
    train, val, fwd, full = split_data(datasets[asset])
    period_slice = {
        "Training": train,
        "Validation": val,
        "Forward Test": fwd,
        "Full History": full,
    }[period_name]

    chosen_strat = best_strategy_lookup.get(asset, "Human")
    bt_period = run_backtest(strategy_map[chosen_strat](period_slice))
    strat_returns[asset] = bt_period["Strategy_Return"].fillna(0)

    calib_df = (
        pd.concat([train, val])
        if period_name == "Forward Test" and len(val) > 20
        else train
    )
    bt_calib = run_backtest(strategy_map[chosen_strat](calib_df))
    calib_ret = bt_calib["Strategy_Return"].dropna()
    ann_vol = calib_ret.std() * np.sqrt(252)
    ann_ret = calib_ret.mean() * 252
    calib_sharpe = (
        ((ann_ret - RISK_FREE_RATE) / ann_vol) if ann_vol > 1e-6 else 0.0
    )

    inv_vol_raw[asset] = (1.0 / ann_vol) if ann_vol > 1e-6 else 1.0
    ai_score_raw[asset] = inv_vol_raw[asset] * max(0.1, 1.0 + calib_sharpe)

  ret_df = pd.DataFrame(strat_returns).fillna(0)
  n = len(selected_assets)
  w_equal = pd.Series({a: 1.0 / n for a in selected_assets})

  s_inv = pd.Series(inv_vol_raw)
  w_inv_vol = s_inv / s_inv.sum()

  s_ai = pd.Series(ai_score_raw)
  w_ai = s_ai / s_ai.sum()

  weights_df = pd.DataFrame({
      "Asset": selected_assets,
      "Best Strategy": [
          best_strategy_lookup.get(a, "Human") for a in selected_assets
      ],
      "Equal Weight (%)": (w_equal * 100).round(2).values,
      "Inverse-Volatility Weight (%)": (w_inv_vol * 100).round(2).values,
      "AI-Suggested Weight (%)": (w_ai * 100).round(2).values,
  })

  curves = {}
  summary_rows = []
  for scheme_name, w_vec in [
      ("Equal Weight", w_equal),
      ("Inverse-Volatility Weight", w_inv_vol),
      ("AI-Suggested Weight", w_ai),
  ]:
    p_ret = ret_df.mul(w_vec, axis=1).sum(axis=1)
    curve = INITIAL_CAPITAL * (1 + p_ret).cumprod()
    curves[scheme_name] = curve

    if len(curve) >= 2:
      tot_ret = curve.iloc[-1] / curve.iloc[0] - 1
      ann_ret = (1 + tot_ret) ** (252 / len(curve)) - 1
      ann_vol = p_ret.std() * np.sqrt(252)
      sharpe = (
          ((p_ret.mean() * 252 - RISK_FREE_RATE) / ann_vol)
          if ann_vol > 1e-6
          else 0.0
      )
      peak = curve.cummax()
      dd = (curve - peak) / peak
      max_dd = dd.min()
    else:
      tot_ret = ann_ret = ann_vol = sharpe = max_dd = np.nan

    summary_rows.append({
        "Weighting Scheme": scheme_name,
        "Period": period_name,
        "Total Return %": round(tot_ret * 100, 2),
        "Annualised Return %": round(ann_ret * 100, 2),
        "Annualised Volatility %": round(ann_vol * 100, 2),
        "Max Drawdown %": round(max_dd * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
    })

  return weights_df, pd.DataFrame(summary_rows), curves, ret_df


# ==========================================================
# STREAMLIT UI & SIDEBAR CONTROLS
# ==========================================================
st.title("STQA3013 Interactive Investment & Strategy Monitoring Dashboard")
st.caption(
    f"Initial Capital: RM {INITIAL_CAPITAL:,} | Risk-Free Rate:"
    f" {RISK_FREE_RATE*100:.1f}% | Transaction Cost:"
    f" {TRANSACTION_COST*100:.2f}% (Next-Day Execution)"
)

datasets = load_datasets()
if not datasets:
  st.error(
      "No CSV files found in the repository root. Please check your uploaded CSV"
      " files."
  )
  st.stop()

st.sidebar.header("Dashboard Controls")

period_choice = st.sidebar.selectbox(
    "1. Evaluation Period",
    options=["Forward Test", "Validation", "Training", "Full History"],
    index=0,
)

asset_choice = st.sidebar.selectbox(
    "2. Single-Asset View",
    options=list(datasets.keys()),
    index=0,
)

strategy_choice = st.sidebar.selectbox(
    "3. Trading Strategy",
    options=["Human", "AI", "Benchmark"],
    format_func=lambda x: {
        "Human": "Human (MA50 > MA200 Crossover)",
        "AI": "AI (RSI < 35 & MACD > 0)",
        "Benchmark": "Benchmark (Buy & Hold)",
    }[x],
    index=0,
)

dd_limit = st.sidebar.slider(
    "4. Max Drawdown Risk Alert Limit (%)",
    min_value=5.0,
    max_value=50.0,
    value=15.0,
    step=1.0,
    help=(
        "Flags a risk alert if maximum drawdown breaches this negative threshold"
        " (e.g., 15% = -15.0% drawdown)."
    ),
)

graph_choices = st.sidebar.multiselect(
    "5. Optional Charts to Display",
    options=[
        "Price & Trading Signals",
        "Cumulative Growth Comparison",
        "Drawdown Curve",
        "Forward-Test vs. Backtest Tracker",
        "3-Scheme Portfolio Comparison",
        "Multi-Asset Price Trends",
        "Daily Return Distribution",
        "Cumulative Return Comparison (All Assets)",
        "Asset Correlation Heatmap",
    ],
    default=[
        "Price & Trading Signals",
        "Cumulative Growth Comparison",
        "Drawdown Curve",
        "Forward-Test vs. Backtest Tracker",
        "3-Scheme Portfolio Comparison",
    ],
)

selected_portfolio_assets = st.sidebar.multiselect(
    "6. Assets Included in Portfolio & Multi-Asset Views",
    options=list(datasets.keys()),
    default=list(datasets.keys()),
)

strategy_map = {
    "Human": human_strategy,
    "AI": ai_strategy,
    "Benchmark": benchmark_strategy,
}

strategy_label_map = {
    "Human": "MA 50/200 crossover",
    "AI": "RSI 35 & MACD > 0",
    "Benchmark": "buy and hold",
}


def get_period_slice(df, period_name):
  train, validation, forward, full = split_data(df)
  return {
      "Training": train,
      "Validation": validation,
      "Forward Test": forward,
      "Full History": full,
  }[period_name]


# ==========================================================
# PRECOMPUTE ALL ASSET-STRATEGY-PERIOD METRICS
# ==========================================================
all_results = []
for asset, df in datasets.items():
  train, validation, forward, _ = split_data(df)
  for s_name, s_fn in strategy_map.items():
    for p_name, p_df in [
        ("Training", train),
        ("Validation", validation),
        ("Forward Test", forward),
    ]:
      res = run_backtest(s_fn(p_df))
      m = performance_metrics(res, is_benchmark=(s_name == "Benchmark"))
      m["Asset"] = asset
      m["Strategy"] = s_name
      m["Period"] = p_name
      all_results.append(m)

results_df = pd.DataFrame(all_results)

val_best = (
    results_df[results_df["Period"] == "Validation"]
    .sort_values("Sharpe", ascending=False)
    .groupby("Asset")
    .first()
)
best_strategy_per_asset = val_best["Strategy"].to_dict()

# ==========================================================
# SINGLE-ASSET KPI CARDS & RISK ALERT BANNER
# ==========================================================
period_df = get_period_slice(datasets[asset_choice], period_choice)
bt_df = run_backtest(strategy_map[strategy_choice](period_df))
metrics = performance_metrics(bt_df, is_benchmark=(strategy_choice == "Benchmark"))

st.subheader(f"{asset_choice} — {strategy_choice} Strategy ({period_choice})")

max_dd_val = metrics["Max DD %"]
if pd.notna(max_dd_val) and max_dd_val < -dd_limit:
  st.error(
      f"🚨 RISK ALERT: {asset_choice} ({strategy_choice}, {period_choice})"
      f" Maximum Drawdown is {max_dd_val:.2f}%, breaching the -{dd_limit:.1f}%"
      " risk limit!"
  )
elif pd.notna(max_dd_val):
  st.success(
      f"✅ WITHIN RISK LIMIT: {asset_choice} ({strategy_choice},"
      f" {period_choice}) Maximum Drawdown is {max_dd_val:.2f}% (limit:"
      f" -{dd_limit:.1f}%)."
  )

c1, c2, c3, c4, c5, c6 = st.columns(6)
end_val = (
    bt_df["Portfolio"].dropna().iloc[-1]
    if len(bt_df["Portfolio"].dropna())
    else np.nan
)
c1.metric("Ending Value", f"RM {end_val:,.2f}" if pd.notna(end_val) else "N/A")
c2.metric("Total Return", f"{metrics['Return %']}%")
c3.metric("Annualised Return", f"{metrics['Annual Return %']}%")
c4.metric("Max Drawdown", f"{metrics['Max DD %']}%")
c5.metric("Sharpe Ratio", f"{metrics['Sharpe']}")
win_rate_display = (
    "N/A" if pd.isna(metrics["Win Rate %"]) else f"{metrics['Win Rate %']}%"
)
c6.metric(
    "Trades | Win Rate",
    f"{metrics['Trades']} | {win_rate_display}",
)

# ==========================================================
# INTERACTIVE PLOTLY CHARTS (HOVER TO SEE DATE & VALUE)
# ==========================================================
if "Price & Trading Signals" in graph_choices:
  st.markdown(
      f"### Price & Trading Signals ({asset_choice} — {period_choice})"
  )
  buy = (bt_df["Position"] == 1) & (bt_df["Position"].shift(1).fillna(0) == 0)
  sell = (bt_df["Position"] == 0) & (bt_df["Position"].shift(1).fillna(0) == 1)

  fig = go.Figure()
  fig.add_trace(
      go.Scatter(
          x=bt_df.index,
          y=bt_df["Price"],
          mode="lines",
          name="Price",
          line=dict(color="black", width=1.8),
          hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: %{y:,.4f}<extra></extra>",
      )
  )
  if strategy_choice == "Human":
    fig.add_trace(
        go.Scatter(
            x=bt_df.index,
            y=bt_df["MA50"],
            mode="lines",
            name="MA50",
            line=dict(color="royalblue", width=1.3, dash="dash"),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>MA50: %{y:,.4f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bt_df.index,
            y=bt_df["MA200"],
            mode="lines",
            name="MA200",
            line=dict(color="darkorange", width=1.3, dash="dash"),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>MA200: %{y:,.4f}<extra></extra>"
            ),
        )
    )
  fig.add_trace(
      go.Scatter(
          x=bt_df.index[buy],
          y=bt_df["Price"][buy],
          mode="markers",
          name="BUY",
          marker=dict(symbol="triangle-up", color="green", size=12),
          hovertemplate=(
              "BUY Signal<br>Date: %{x|%Y-%m-%d}<br>Price:"
              " %{y:,.4f}<extra></extra>"
          ),
      )
  )
  fig.add_trace(
      go.Scatter(
          x=bt_df.index[sell],
          y=bt_df["Price"][sell],
          mode="markers",
          name="SELL",
          marker=dict(symbol="triangle-down", color="red", size=12),
          hovertemplate=(
              "SELL Signal<br>Date: %{x|%Y-%m-%d}<br>Price:"
              " %{y:,.4f}<extra></extra>"
          ),
      )
  )
  fig.update_layout(
      title=(
          f"{asset_choice} ({period_choice}): Price and {strategy_choice}"
          " Trading Signals"
      ),
      xaxis_title="Date",
      yaxis_title="Price",
      hovermode="x unified",
      height=460,
      template="plotly_white",
  )
  st.plotly_chart(fig, use_container_width=True)

if "Cumulative Growth Comparison" in graph_choices:
  st.markdown(
      f"### Equity Curve vs. Buy-and-Hold ({asset_choice} — {period_choice})"
  )
  strategy_growth = (
      1 + bt_df["Strategy_Return"].fillna(0)
  ).cumprod() * INITIAL_CAPITAL
  benchmark_growth = (1 + bt_df["Return"].fillna(0)).cumprod() * INITIAL_CAPITAL

  fig = go.Figure()
  fig.add_trace(
      go.Scatter(
          x=strategy_growth.index,
          y=strategy_growth,
          mode="lines",
          name=f"{strategy_choice} Strategy (RM)",
          line=dict(color="royalblue", width=2.2),
          hovertemplate=(
              "Date: %{x|%Y-%m-%d}<br>Strategy: RM %{y:,.2f}<extra></extra>"
          ),
      )
  )
  fig.
