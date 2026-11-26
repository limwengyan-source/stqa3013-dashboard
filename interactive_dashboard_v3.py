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

GROUP_ID = "G01"
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
    # Next-day execution to avoid look-ahead bias
    df["Position"] = df["Signal"].shift(1).fillna(0)
    trades = df["Position"].diff().abs().fillna(0)
    cost = trades * TRANSACTION_COST
    df["Strategy_Return"] = df["Position"] * df["Return"] - cost
    df["Portfolio"] = INITIAL_CAPITAL * (1 + df["Strategy_Return"].fillna(0)).cumprod()
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

    # Count entries (round-trip trades / buy entries; 1 for buy-and-hold benchmark)
    if is_benchmark:
        num_trades = 1
        win_rate_val = np.nan
    else:
        entries = ((df["Position"] == 1) & (df["Position"].shift(1).fillna(0) == 0)).sum()
        num_trades = int(entries)
        wins = (daily > 0).sum()
        losses = (daily < 0).sum()
        active_days = wins + losses
        win_rate_val = round((wins / active_days) * 100, 2) if active_days > 0 else 0.0

    return {
        "Return %": round(total_return * 100, 2),
        "Annual Return %": round(annual_return * 100, 2),
        "Sharpe": round(sharpe, 2),
        "Max DD %": round(max_drawdown * 100, 2),
        "Trades": num_trades,
        "Win Rate %": win_rate_val,
    }


def build_portfolio_schemes(datasets, selected_assets, period_name, best_strategy_lookup):
    """
    Task C: Combine best strategy per asset into one portfolio and compare 3 weighting schemes:
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

        # Estimate weights from pre-forward data (Training + Validation) so forward test stays out-of-sample
        calib_df = pd.concat([train, val]) if period_name == "Forward Test" and len(val) > 20 else train
        bt_calib = run_backtest(strategy_map[chosen_strat](calib_df))
        calib_ret = bt_calib["Strategy_Return"].dropna()
        ann_vol = calib_ret.std() * np.sqrt(252)
        ann_ret = calib_ret.mean() * 252
        calib_sharpe = ((ann_ret - RISK_FREE_RATE) / ann_vol) if ann_vol > 1e-6 else 0.0

        inv_vol_raw[asset] = (1.0 / ann_vol) if ann_vol > 1e-6 else 1.0
        # AI-suggested logic: combine inverse volatility with positive Sharpe tilt (floor at 0.1 so all assets keep diversification exposure)
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
        "Best Strategy": [best_strategy_lookup.get(a, "Human") for a in selected_assets],
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
            sharpe = ((p_ret.mean() * 252 - RISK_FREE_RATE) / ann_vol) if ann_vol > 1e-6 else 0.0
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
    f"Group {GROUP_ID} | Initial Capital: RM {INITIAL_CAPITAL:,} | Risk-Free Rate: {RISK_FREE_RATE*100:.1f}% | "
    f"Transaction Cost: {TRANSACTION_COST*100:.2f}% (Next-Day Execution)"
)

datasets = load_datasets()
if not datasets:
    st.error("No CSV files found in the repository root. Please check your uploaded CSV files.")
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
    help="Flags a risk alert if maximum drawdown breaches this negative threshold (e.g., 15% = -15.0% drawdown).",
)

graph_choices = st.sidebar.multiselect(
    "5. Optional Charts to Display",
    options=[
        "Price & Trading Signals",
        "Cumulative Growth Comparison",
        "Drawdown Curve",
        "Forward-Test vs. Backtest Tracker",
        "Task C: 3-Scheme Portfolio Comparison",
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
        "Task C: 3-Scheme Portfolio Comparison",
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
        for p_name, p_df in [("Training", train), ("Validation", validation), ("Forward Test", forward)]:
            res = run_backtest(s_fn(p_df))
            m = performance_metrics(res, is_benchmark=(s_name == "Benchmark"))
            m["Asset"] = asset
            m["Strategy"] = s_name
            m["Period"] = p_name
            all_results.append(m)

results_df = pd.DataFrame(all_results)

# Best strategy per asset selected strictly from Validation (before locking for Forward Test)
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

# Task D: Risk Alert Flag when Max Drawdown breaches threshold
max_dd_val = metrics["Max DD %"]
if pd.notna(max_dd_val) and max_dd_val < -dd_limit:
    st.error(
        f"🚨 RISK ALERT: {asset_choice} ({strategy_choice}, {period_choice}) Maximum Drawdown is "
        f"{max_dd_val:.2f}%, breaching the -{dd_limit:.1f}% risk limit!"
    )
elif pd.notna(max_dd_val):
    st.success(
        f"✅ WITHIN RISK LIMIT: {asset_choice} ({strategy_choice}, {period_choice}) Maximum Drawdown is "
        f"{max_dd_val:.2f}% (limit: -{dd_limit:.1f}%)."
    )

c1, c2, c3, c4, c5, c6 = st.columns(6)
end_val = bt_df["Portfolio"].dropna().iloc[-1] if len(bt_df["Portfolio"].dropna()) else np.nan
c1.metric("Ending Value", f"RM {end_val:,.2f}" if pd.notna(end_val) else "N/A")
c2.metric("Total Return", f"{metrics['Return %']}%")
c3.metric("Annualised Return", f"{metrics['Annual Return %']}%")
c4.metric("Max Drawdown", f"{metrics['Max DD %']}%")
c5.metric("Sharpe Ratio", f"{metrics['Sharpe']}")
c6.metric(
    "Trades | Win Rate",
    f"{metrics['Trades']} | {'N/A' if pd.isna(metrics['Win Rate %']) else str(metrics['Win Rate %']) + '%'}",
)

# ==========================================================
# OPTIONAL CHARTS
# ==========================================================
if "Price & Trading Signals" in graph_choices:
    st.markdown(f"### Price & Trading Signals ({asset_choice} — {period_choice})")
    buy = (bt_df["Position"] == 1) & (bt_df["Position"].shift(1).fillna(0) == 0)
    sell = (bt_df["Position"] == 0) & (bt_df["Position"].shift(1).fillna(0) == 1)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(bt_df.index, bt_df["Price"], label="Price", color="black", linewidth=1.2)
    if strategy_choice == "Human":
        ax.plot(bt_df.index, bt_df["MA50"], "--", label="MA50", color="royalblue", alpha=0.85)
        ax.plot(bt_df.index, bt_df["MA200"], "--", label="MA200", color="darkorange", alpha=0.85)
    ax.scatter(bt_df.index[buy], bt_df["Price"][buy], marker="^", color="green", s=90, label="BUY", zorder=5)
    ax.scatter(bt_df.index[sell], bt_df["Price"][sell], marker="v", color="red", s=90, label="SELL", zorder=5)
    ax.set_title(f"{asset_choice} ({period_choice}): Price and {strategy_choice} Trading Signals")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

if "Cumulative Growth Comparison" in graph_choices:
    st.markdown(f"### Equity Curve vs. Buy-and-Hold ({asset_choice} — {period_choice})")
    strategy_growth = (1 + bt_df["Strategy_Return"].fillna(0)).cumprod() * INITIAL_CAPITAL
    benchmark_growth = (1 + bt_df["Return"].fillna(0)).cumprod() * INITIAL_CAPITAL
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(strategy_growth.index, strategy_growth, label=f"{strategy_choice} Strategy (RM)", linewidth=2)
    ax.plot(benchmark_growth.index, benchmark_growth, "--", label="Buy & Hold Benchmark (RM)", linewidth=1.5)
    ax.set_title(f"{asset_choice} ({period_choice}): Equity Curve Comparison (Start = RM {INITIAL_CAPITAL:,})")
    ax.set_ylabel("Portfolio Value (RM)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

if "Drawdown Curve" in graph_choices:
    st.markdown(f"### Drawdown Chart ({asset_choice} — {period_choice})")
    growth = (1 + bt_df["Strategy_Return"].fillna(0)).cumprod()
    rolling_max = growth.cummax()
    drawdown = (growth - rolling_max) / rolling_max
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.fill_between(drawdown.index, drawdown * 100, 0, color="red", alpha=0.3)
    ax.plot(drawdown.index, drawdown * 100, color="darkred", label="Drawdown (%)")
    ax.axhline(-dd_limit, color="black", linestyle="--", linewidth=1.2, label=f"Risk Alert Limit (-{dd_limit:.1f}%)")
    ax.set_title(f"{asset_choice} ({period_choice}): {strategy_choice} Drawdown Curve")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

# Task D: Forward-Test Tracker (1 Jan - 30 Sep 2026) vs Backtest (Training & Validation)
if "Forward-Test vs. Backtest Tracker" in graph_choices:
    st.markdown(f"### Forward-Test Tracker (1 Jan – 30 Sep 2026) vs. Backtest ({asset_choice} — {strategy_choice})")
    tracker_df = results_df[
        (results_df["Asset"] == asset_choice) & (results_df["Strategy"] == strategy_choice)
    ][["Period", "Return %", "Annual Return %", "Max DD %", "Sharpe", "Trades", "Win Rate %"]]

    t_col1, t_col2 = st.columns([1.1, 1.3])
    with t_col1:
        st.dataframe(tracker_df, use_container_width=True, hide_index=True)
    with t_col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(tracker_df))
        width = 0.35
        ax.bar(x - width / 2, tracker_df["Annual Return %"], width, label="Annualised Return (%)", color="steelblue")
        ax.bar(x + width / 2, tracker_df["Max DD %"], width, label="Max Drawdown (%)", color="indianred")
        ax.set_xticks(x)
        ax.set_xticklabels(tracker_df["Period"])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{asset_choice} ({strategy_choice}): Backtest vs. Forward Test 2026")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        st.pyplot(fig, clear_figure=True)

# Task C: Three Portfolio Weighting Schemes
if "Task C: 3-Scheme Portfolio Comparison" in graph_choices and len(selected_portfolio_assets) >= 2:
    st.markdown(f"### Task C: Multi-Asset Portfolio Comparison ({period_choice})")
    st.caption(
        "Combines each asset's Validation-selected best strategy under three weighting schemes: "
        "(1) Equal Weight, (2) Inverse-Volatility Weight (1/σ estimated pre-forward), and "
        "(3) AI-Suggested Weight (Inverse-Volatility tilted by positive pre-forward Sharpe ratio)."
    )

    weights_df, port_summary_df, port_curves, port_ret_df = build_portfolio_schemes(
        datasets, selected_portfolio_assets, period_choice, best_strategy_per_asset
    )

    p_col1, p_col2 = st.columns([1, 1.3])
    with p_col1:
        st.markdown("**Asset Allocation Weights (%)**")
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
    with p_col2:
        st.markdown("**Portfolio Performance Across Weighting Schemes**")
        st.dataframe(port_summary_df, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    for scheme_name, curve in port_curves.items():
        ax.plot(curve.index, curve, linewidth=2, label=scheme_name)
    ax.set_title(f"Portfolio Equity Curves Across 3 Weighting Schemes ({period_choice})")
    ax.set_ylabel("Portfolio Value (RM)")
    ax.grid(True, alpha=0.3)
    ax.legend()
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

# ==========================================================
# FULL METRICS TABLE & CLASS SUBMISSION CSV DOWNLOAD
# ==========================================================
st.markdown("---")
st.subheader(f"Performance Metrics Table ({period_choice})")

if period_choice in ["Training", "Validation", "Forward Test"]:
    filtered_summary = results_df[results_df["Period"] == period_choice][
        ["Asset", "Strategy", "Return %", "Annual Return %", "Max DD %", "Sharpe", "Trades", "Win Rate %"]
    ]
    st.dataframe(filtered_summary, use_container_width=True, hide_index=True)
else:
    st.dataframe(
        results_df[["Asset", "Period", "Strategy", "Return %", "Annual Return %", "Max DD %", "Sharpe", "Trades", "Win Rate %"]],
        use_container_width=True,
        hide_index=True,
    )

# Build exact 11-column class submission CSV (including portfolio rows for Task C)
submission_rows = []
period_csv_map = {"Training": "training", "Validation": "validation", "Forward Test": "forward"}

for _, row in results_df.iterrows():
    s_type = row["Strategy"].lower()
    submission_rows.append({
        "group_id": GROUP_ID,
        "asset": row["Asset"],
        "strategy_type": s_type,
        "strategy_name": strategy_label_map[row["Strategy"]],
        "period": period_csv_map[row["Period"]],
        "total_return_pct": row["Return %"],
        "annualised_return_pct": row["Annual Return %"],
        "max_drawdown_pct": row["Max DD %"],
        "sharpe": row["Sharpe"],
        "trades": row["Trades"],
        "win_rate_pct": "" if s_type == "benchmark" or pd.isna(row["Win Rate %"]) else row["Win Rate %"],
    })

if len(selected_portfolio_assets) >= 2:
    for p_label, p_csv in period_csv_map.items():
        _, p_sum_df, _, _ = build_portfolio_schemes(
            datasets, selected_portfolio_assets, p_label, best_strategy_per_asset
        )
        for _, prow in p_sum_df.iterrows():
            w_name = prow["Weighting Scheme"]
            w_type = "ai" if "AI" in w_name else ("human" if "Inverse" in w_name else "benchmark")
            submission_rows.append({
                "group_id": GROUP_ID,
                "asset": "portfolio",
                "strategy_type": w_type,
                "strategy_name": w_name,
                "period": p_csv,
                "total_return_pct": prow["Total Return %"],
                "annualised_return_pct": prow["Annualised Return %"],
                "max_drawdown_pct": prow["Max Drawdown %"],
                "sharpe": prow["Sharpe Ratio"],
                "trades": "",
                "win_rate_pct": "",
            })

submission_df = pd.DataFrame(submission_rows)
st.download_button(
    label="Download Class Submission Results CSV",
    data=submission_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{GROUP_ID}_STQA3013_results.csv",
    mime="text/csv",
)
