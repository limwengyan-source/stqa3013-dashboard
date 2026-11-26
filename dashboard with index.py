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
      tot_
