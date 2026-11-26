# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
# DATA LOADING & CLEANING (EXACT LOGIC FROM YOUR NOTEBOOK)
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
            df["Price"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
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
    df["Position"] = df["Signal"].shift(1).fillna(0)
    trades = df["Position"].diff().abs().fillna(0)
    cost = trades * TRANSACTION_COST
    df["Strategy_Return"] = df["Position"] * df["Return"] - cost
    df["Portfolio"] = INITIAL_CAPITAL * (1 + df["Strategy_Return"].fillna(0)).cumprod()
    return df


def performance_metrics(df):
    portfolio = df["Portfolio"].dropna()
    if len(portfolio) < 2:
        return {
            "Return %": np.nan,
            "Annual Return %": np.nan,
            "Sharpe": np.nan,
            "Max DD %": np.nan,
            "Win Rate %": np.nan,
        }
    total_return = portfolio.iloc[-1] / portfolio.iloc[0] - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio)) - 1
    daily = portfolio.pct_change().dropna()
    annual_volatility = daily.std() * np.sqrt(252)
    if np.isnan(annual_volatility) or annual_volatility < 0.05:
        sharpe = 0
    else:
        sharpe = (daily.mean() * 252 - RISK_FREE_RATE) / annual_volatility
    sharpe = max(min(sharpe, 5), -5)

    rolling_peak = portfolio.cummax()
    drawdown = (portfolio - rolling_peak) / rolling_peak
    max_drawdown = drawdown.min()

    wins = (daily > 0).sum()
    losses = (daily < 0).sum()
    total_trades = wins + losses
    win_rate = 0 if total_trades == 0 else (wins / total_trades)

    return {
        "Return %": round(total_return * 100, 2),
        "Annual Return %": round(annual_return * 100, 2),
        "Sharpe": round(sharpe, 2),
        "Max DD %": round(max_drawdown * 100, 2),
        "Win Rate %": round(win_rate * 100, 2),
    }


# ==========================================================
# STREAMLIT INTERACTIVE CONTROLS
# ==========================================================
st.title("STQA3013 Interactive Investment & Strategy Dashboard")
st.caption(
    f"Initial Capital: RM {INITIAL_CAPITAL:,} | Risk-Free Rate: {RISK_FREE_RATE*100:.1f}% | "
    f"Transaction Cost: {TRANSACTION_COST*100:.2f}%"
)

datasets = load_datasets()
if not datasets:
    st.error("No CSV files found in the repository root. Please check your uploaded CSV files.")
    st.stop()

st.sidebar.header("Dashboard Filters")

period_choice = st.sidebar.selectbox(
    "1. Select Evaluation Period",
    options=["Forward Test", "Validation", "Training", "Full History"],
    index=0,
)

asset_choice = st.sidebar.selectbox(
    "2. Select Asset (Single-Asset View)",
    options=list(datasets.keys()),
    index=0,
)

strategy_choice = st.sidebar.selectbox(
    "3. Select Trading Strategy",
    options=["Human", "AI", "Benchmark"],
    format_func=lambda x: {
        "Human": "Human (MA50 > MA200)",
        "AI": "AI (RSI < 35 & MACD > 0)",
        "Benchmark": "Benchmark (Buy & Hold)",
    }[x],
    index=0,
)

graph_choices = st.sidebar.multiselect(
    "4. Select Graphs to Display",
    options=[
        "Price & Trading Signals",
        "Cumulative Growth Comparison",
        "Drawdown Curve",
        "Multi-Asset Price Trends",
        "Daily Return Distribution",
        "Cumulative Return Comparison (All Assets)",
        "Asset Correlation Heatmap",
        "Equal Weight Portfolio & Drawdown",
    ],
    default=[
        "Price & Trading Signals",
        "Cumulative Growth Comparison",
        "Drawdown Curve",
        "Equal Weight Portfolio & Drawdown",
    ],
)

selected_portfolio_assets = st.sidebar.multiselect(
    "5. Assets Included in Multi-Asset & Portfolio Charts",
    options=list(datasets.keys()),
    default=list(datasets.keys()),
)

strategy_map = {
    "Human": human_strategy,
    "AI": ai_strategy,
    "Benchmark": benchmark_strategy,
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
# COMPUTE RESULTS FOR SELECTED PERIOD & ALL PERIODS
# ==========================================================
all_results = []
for asset, df in datasets.items():
    train, validation, forward, _ = split_data(df)
    for s_name, s_fn in strategy_map.items():
        for p_name, p_df in [("Training", train), ("Validation", validation), ("Forward Test", forward)]:
            res = run_backtest(s_fn(p_df))
            m = performance_metrics(res)
            m["Asset"] = asset
            m["Strategy"] = s_name
            m["Period"] = p_name
            all_results.append(m)

results_df = pd.DataFrame(all_results)

# Single-asset selected run
period_df = get_period_slice(datasets[asset_choice], period_choice)
bt_df = run_backtest(strategy_map[strategy_choice](period_df))
metrics = performance_metrics(bt_df)

st.subheader(f"{asset_choice} — {strategy_choice} Strategy ({period_choice})")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ending Portfolio", f"RM {bt_df['Portfolio'].dropna().iloc[-1]:,.2f}" if len(bt_df["Portfolio"].dropna()) else "N/A")
c2.metric("Return %", f"{metrics['Return %']}%")
c3.metric("Annual Return %", f"{metrics['Annual Return %']}%")
c4.metric("Sharpe Ratio", f"{metrics['Sharpe']}")
c5.metric("Max DD % | Win Rate %", f"{metrics['Max DD %']}% | {metrics['Win Rate %']}%")

# ==========================================================
# OPTIONAL GRAPHS BASED ON SIDEBAR SELECTION
# ==========================================================
if "Price & Trading Signals" in graph_choices:
    st.markdown(f"### Price & Trading Signals ({asset_choice} - {period_choice})")
    buy = (bt_df["Position"] == 1) & (bt_df["Position"].shift(1) == 0)
    sell = (bt_df["Position"] == 0) & (bt_df["Position"].shift(1) == 1)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(bt_df.index, bt_df["Price"], label="Price", color="black")
    if strategy_choice == "Human":
        ax.plot(bt_df.index, bt_df["MA50"], "--", label="MA50", color="royalblue", alpha=0.8)
        ax.plot(bt_df.index, bt_df["MA200"], "--", label="MA200", color="darkorange", alpha=0.8)
    ax.scatter(bt_df.index[buy], bt_df["Price"][buy], marker="^", color="green", s=85, label="BUY", zorder=5)
    ax.scatter(bt_df.index[sell], bt_df["Price"][sell], marker="v", color="red", s=85, label="SELL", zorder=5)
    ax.set_title(f"{asset_choice} ({period_choice}): Price and {strategy_choice} Trading Signals")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

if "Cumulative Growth Comparison" in graph_choices:
    st.markdown(f"### Cumulative Growth Comparison ({asset_choice} - {period_choice})")
    strategy_growth = (1 + bt_df["Strategy_Return"].fillna(0)).cumprod()
    benchmark_growth = (1 + bt_df["Return"].fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(strategy_growth.index, strategy_growth, label=f"{strategy_choice} Strategy", linewidth=2)
    ax.plot(benchmark_growth.index, benchmark_growth, "--", label="Buy & Hold", linewidth=1.5)
    ax.set_title(f"{asset_choice} ({period_choice}): Cumulative Growth Comparison (Base = 1)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

if "Drawdown Curve" in graph_choices:
    st.markdown(f"### Drawdown Curve ({asset_choice} - {period_choice})")
    growth = (1 + bt_df["Strategy_Return"].fillna(0)).cumprod()
    rolling_max = growth.cummax()
    drawdown = (growth - rolling_max) / rolling_max
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.fill_between(drawdown.index, drawdown * 100, 0, color="red", alpha=0.3)
    ax.plot(drawdown.index, drawdown * 100, color="darkred")
    ax.set_title(f"{asset_choice} ({period_choice}): {strategy_choice} Drawdown Curve")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

if "Multi-Asset Price Trends" in graph_choices and selected_portfolio_assets:
    st.markdown(f"### Multi-Asset Price Trends ({period_choice})")
    n = len(selected_portfolio_assets)
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(16, 4 * rows))
    axes = np.array(axes).flatten()
    for ax, asset in zip(axes, selected_portfolio_assets):
        sub = get_period_slice(datasets[asset], period_choice)
        ax.plot(sub.index, sub["Price"], color="royalblue", linewidth=1.2)
        ax.set_title(f"{asset} Price Trend ({period_choice})")
        ax.grid(True, alpha=0.3)
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

if "Daily Return Distribution" in graph_choices and selected_portfolio_assets:
    st.markdown(f"### Daily Return Distribution ({period_choice})")
    n = len(selected_portfolio_assets)
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(16, 4 * rows))
    axes = np.array(axes).flatten()
    for ax, asset in zip(axes, selected_portfolio_assets):
        sub = get_period_slice(datasets[asset], period_choice)
        rets = sub["Price"].pct_change().dropna()
        sns.histplot(rets, bins=40, kde=True, ax=ax, color="skyblue")
        ax.set_title(f"{asset} Daily Return Distribution ({period_choice})")
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

if "Cumulative Return Comparison (All Assets)" in graph_choices and selected_portfolio_assets:
    st.markdown(f"### Cumulative Return Comparison Across Selected Assets ({period_choice})")
    fig, ax = plt.subplots(figsize=(14, 6))
    for asset in selected_portfolio_assets:
        sub = get_period_slice(datasets[asset], period_choice)
        if len(sub) > 0:
            norm = sub["Price"] / sub["Price"].iloc[0]
            ax.plot(norm.index, norm, label=asset)
    ax.set_title(f"Cumulative Return Comparison ({period_choice})")
    ax.set_ylabel("Growth (Base = 1)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

if "Asset Correlation Heatmap" in graph_choices and len(selected_portfolio_assets) >= 2:
    st.markdown(f"### Asset Return Correlation Heatmap ({period_choice})")
    ret_df = pd.DataFrame()
    for asset in selected_portfolio_assets:
        sub = get_period_slice(datasets[asset], period_choice)
        ret_df[asset] = sub["Price"].pct_change()
    corr = ret_df.corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap="RdYlGn", center=0, linewidths=0.5, fmt=".2f", ax=ax)
    ax.set_title(f"Correlation Heatmap ({period_choice})")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

if "Equal Weight Portfolio & Drawdown" in graph_choices and selected_portfolio_assets:
    st.markdown(f"### Equal Weight Portfolio ({strategy_choice} Strategy — {period_choice})")
    port_rets = {}
    for asset in selected_portfolio_assets:
        sub = get_period_slice(datasets[asset], period_choice)
        bt_sub = run_backtest(strategy_map[strategy_choice](sub))
        port_rets[asset] = bt_sub["Strategy_Return"]
    port_df = pd.DataFrame(port_rets)
    eq_weight = port_df.mean(axis=1)
    eq_curve = INITIAL_CAPITAL * (1 + eq_weight.fillna(0)).cumprod()

    if len(eq_curve.dropna()) >= 2:
        p_ret = eq_curve.iloc[-1] / eq_curve.iloc[0] - 1
        p_daily = eq_curve.pct_change().dropna()
        p_sharpe = ((p_daily.mean() * 252 - RISK_FREE_RATE) / (p_daily.std() * np.sqrt(252))) if p_daily.std() > 0 else 0
        run_max = eq_curve.cummax()
        p_dd = (eq_curve - run_max) / run_max
        p_max_dd = p_dd.min()

        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Portfolio Return (%)", f"{round(p_ret * 100, 2)}%")
        pc2.metric("Portfolio Sharpe Ratio", f"{round(p_sharpe, 2)}")
        pc3.metric("Portfolio Max Drawdown (%)", f"{round(p_max_dd * 100, 2)}%")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        ax1.plot(eq_curve.index, eq_curve, color="navy", linewidth=1.8)
        ax1.set_title(f"Equal Weight Portfolio Equity Curve ({strategy_choice} — {period_choice})")
        ax1.set_ylabel("Portfolio Value (RM)")
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(p_dd.index, p_dd * 100, 0, color="red", alpha=0.35)
        ax2.plot(p_dd.index, p_dd * 100, color="darkred", linewidth=1)
        ax2.set_title("Portfolio Drawdown (%)")
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

# ==========================================================
# PERFORMANCE SUMMARY TABLES
# ==========================================================
st.markdown("---")
st.subheader(f"Performance Summary Table ({period_choice})")
if period_choice in ["Training", "Validation", "Forward Test"]:
    filtered_summary = results_df[results_df["Period"] == period_choice][
        ["Asset", "Strategy", "Return %", "Annual Return %", "Max DD %", "Sharpe", "Win Rate %"]
    ]
    st.dataframe(filtered_summary, use_container_width=True, hide_index=True)
else:
    st.dataframe(
        results_df[["Asset", "Period", "Strategy", "Return %", "Annual Return %", "Max DD %", "Sharpe", "Win Rate %"]],
        use_container_width=True,
        hide_index=True,
    )
