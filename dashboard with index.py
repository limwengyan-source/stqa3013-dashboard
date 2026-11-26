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
  validation = df.loc["2025-01-01":"2025-12
