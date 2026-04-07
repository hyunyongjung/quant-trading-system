"""
quant_system.py
===============
A personal quant trading system using MA cross, ML (XGBoost), and SHAP.

Usage:
    run("RTX")           # Full analysis for a single ticker
    daily_monitor()      # Daily signal check for watchlist
    market_analysis()    # S&P500 rebound probability
    backtest("RTX")      # Backtest: ML vs MA vs Buy&Hold

Requirements:
    pip install yfinance xgboost scikit-learn shap matplotlib
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from datetime import date

# =====================================================
# CONFIG
# =====================================================
START      = "2024-01-01"
CAPITAL    = 300
FEE_RATE   = 0.001

WATCHLIST  = {
    "RTX":       "Holding ($197.40)",
    "TSLA":      "TSLQ holding - exit on Golden Cross",
    "032820.KS": "Holding (Woori Technology)",
    "GOOGL":     "Watchlist",
    "META":      "Watchlist",
}

FEATURES   = ["MA_ratio", "RSI", "Volume_ratio", "Return_1d", "Return_5d", "Volatility"]
FEAT_NAMES = {
    "MA_ratio":     "MA Ratio (MA20/MA60)",
    "RSI":          "RSI (Overbought/Oversold)",
    "Volume_ratio": "Volume Ratio",
    "Return_1d":    "1-Day Return",
    "Return_5d":    "5-Day Return",
    "Volatility":   "Volatility",
}

# =====================================================
# DATA & FEATURE ENGINEERING
# =====================================================
def load_data(ticker, start=START):
    """Download price data and compute all features."""
    data = yf.download(ticker, start=start, progress=False)
    data.columns = data.columns.get_level_values(0)

    # Moving averages
    data["MA20"]         = data["Close"].rolling(20).mean()
    data["MA60"]         = data["Close"].rolling(60).mean()
    data["MA_ratio"]     = data["MA20"] / data["MA60"]

    # RSI
    delta                = data["Close"].diff()
    gain                 = delta.clip(lower=0).rolling(14).mean()
    loss                 = (-delta.clip(upper=0)).rolling(14).mean()
    data["RSI"]          = 100 - (100 / (1 + gain / loss))

    # MACD
    exp1                 = data["Close"].ewm(span=12).mean()
    exp2                 = data["Close"].ewm(span=26).mean()
    data["MACD"]         = exp1 - exp2
    data["MACD_signal"]  = data["MACD"].ewm(span=9).mean()

    # Bollinger Bands
    data["BB_mid"]       = data["Close"].rolling(20).mean()
    data["BB_upper"]     = data["BB_mid"] + 2 * data["Close"].rolling(20).std()
    data["BB_lower"]     = data["BB_mid"] - 2 * data["Close"].rolling(20).std()

    # Volume & returns
    data["Volume_ratio"] = data["Volume"] / data["Volume"].rolling(20).mean()
    data["Return_1d"]    = data["Close"].pct_change(1)
    data["Return_5d"]    = data["Close"].pct_change(5)
    data["Volatility"]   = data["Return_1d"].rolling(20).std()

    # ML label: 1 if price is higher 5 days later
    data["Future_return"]= data["Close"].shift(-5) / data["Close"] - 1
    data["Label"]        = (data["Future_return"] > 0).astype(int)

    return data.dropna()


# =====================================================
# ML MODEL
# =====================================================
def train_model(data):
    """Train XGBoost classifier and return model + split index."""
    X     = data[FEATURES]
    y     = data["Label"]
    split = int(len(data) * 0.8)
    model = XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])
    return model, split


# =====================================================
# SIGNAL EVALUATION
# =====================================================
def get_signals(data, ml_prob):
    """Evaluate 6 indicators and return signal dict + verdict."""
    t          = data.iloc[-1]
    price      = float(t["Close"])
    ma20       = float(t["MA20"])
    ma60       = float(t["MA60"])
    rsi        = float(t["RSI"])
    macd       = float(t["MACD"])
    macd_sig   = float(t["MACD_signal"])
    bb_upper   = float(t["BB_upper"])
    bb_lower   = float(t["BB_lower"])
    vol_ratio  = float(t["Volume_ratio"])

    sigs = {}
    sigs["MA Cross"]  = "BUY  ✅ Golden Cross" if ma20 > ma60 else "SELL ❌ Dead Cross"

    if rsi < 30:
        sigs["RSI"]   = f"BUY  ✅ Oversold {rsi:.1f}"
    elif rsi > 70:
        sigs["RSI"]   = f"SELL ❌ Overbought {rsi:.1f}"
    else:
        sigs["RSI"]   = f"NEUT ⚪ Neutral {rsi:.1f}"

    sigs["MACD"]      = "BUY  ✅ Bullish momentum" if macd > macd_sig else "SELL ❌ Bearish momentum"

    if price < bb_lower:
        sigs["Bollinger"] = "BUY  ✅ Below lower band"
    elif price > bb_upper:
        sigs["Bollinger"] = "SELL ❌ Above upper band"
    else:
        sigs["Bollinger"] = "NEUT ⚪ Inside band"

    sigs["Volume"]    = f"BUY  ✅ High volume {vol_ratio:.2f}x" if vol_ratio > 1.2 else f"NEUT ⚪ Normal volume {vol_ratio:.2f}x"
    sigs["ML"]        = f"BUY  ✅ Upside prob {ml_prob*100:.1f}%" if ml_prob > 0.6 else f"SELL ❌ Upside prob {ml_prob*100:.1f}%"

    buy  = sum(1 for v in sigs.values() if "BUY"  in v)
    sell = sum(1 for v in sigs.values() if "SELL" in v)
    neut = sum(1 for v in sigs.values() if "NEUT" in v)

    if   buy  >= 4: verdict = "🟢 STRONG BUY"
    elif buy  == 3: verdict = "🟡 WEAK BUY"
    elif sell >= 4: verdict = "🔴 STRONG SELL / HOLD CASH"
    elif sell == 3: verdict = "🟠 WEAK SELL"
    else:           verdict = "⚪ NEUTRAL"

    return sigs, buy, sell, neut, verdict, price


# =====================================================
# STATS HELPER
# =====================================================
def calc_stats(portfolio, capital):
    """Return final value, return %, MDD %, Sharpe ratio."""
    final      = portfolio[-1]
    ret        = (final - capital) / capital * 100
    peak       = pd.Series(portfolio).cummax()
    mdd        = ((pd.Series(portfolio) - peak) / peak * 100).min()
    daily_ret  = pd.Series(portfolio).pct_change().dropna()
    sharpe     = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
    return final, ret, mdd, sharpe


# =====================================================
# 1. DAILY MONITOR
# =====================================================
def daily_monitor():
    """Check MA signals for all watchlist tickers."""
    print(f"{'='*50}")
    print(f"  Daily Signal Check — {date.today()}")
    print(f"{'='*50}\n")

    for ticker, status in WATCHLIST.items():
        try:
            data      = yf.download(ticker, start=START, progress=False)
            data.columns = data.columns.get_level_values(0)
            data["MA20"] = data["Close"].rolling(20).mean()
            data["MA60"] = data["Close"].rolling(60).mean()

            price     = float(data["Close"].iloc[-1])
            ma20      = float(data["MA20"].iloc[-1])
            ma60      = float(data["MA60"].iloc[-1])
            prev_ma20 = float(data["MA20"].iloc[-2])
            prev_ma60 = float(data["MA60"].iloc[-2])

            signal    = "✅ GOLDEN" if ma20 > ma60 else "❌ DEAD"
            alert     = ""
            if prev_ma20 <= prev_ma60 and ma20 > ma60:
                alert = "  ⚡ GOLDEN CROSS — consider BUY"
            elif prev_ma20 >= prev_ma60 and ma20 < ma60:
                alert = "  🚨 DEAD CROSS — consider SELL"

            print(f"[{status}] {ticker}")
            print(f"  Price: ${price:.2f}  MA20: ${ma20:.2f}  MA60: ${ma60:.2f}  {signal}{alert}")
            print()
        except Exception as e:
            print(f"{ticker}: error — {e}\n")


# =====================================================
# 2. FULL ANALYSIS (signals + SHAP + charts)
# =====================================================
def analyze(ticker):
    """6-indicator analysis + SHAP explanation + price charts."""
    print(f"\n{'='*50}")
    print(f"  Analyzing {ticker} — {date.today()}")
    print(f"{'='*50}")

    data           = load_data(ticker)
    model, split   = train_model(data)
    X              = data[FEATURES]
    ml_prob        = model.predict_proba(X.iloc[[-1]])[0][1]
    sigs, buy, sell, neut, verdict, price = get_signals(data, ml_prob)

    # --- Signal summary ---
    print(f"\n  Price: ${price:.2f}\n")
    print("[Signals]")
    for name, sig in sigs.items():
        print(f"  {name:<12}: {sig}")
    print(f"\n  Buy: {buy}  Sell: {sell}  Neutral: {neut}")
    print(f"  Verdict: {verdict}\n")

    # --- Price / MA / Bollinger chart ---
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    color = "green" if float(data["MA20"].iloc[-1]) > float(data["MA60"].iloc[-1]) else "red"

    axes[0].plot(data["Close"],    label="Price",    alpha=0.7)
    axes[0].plot(data["MA20"],     label="MA20",     linewidth=1.5)
    axes[0].plot(data["MA60"],     label="MA60",     linewidth=1.5)
    axes[0].plot(data["BB_upper"], label="BB Upper", linestyle="--", color="gray", alpha=0.6)
    axes[0].plot(data["BB_lower"], label="BB Lower", linestyle="--", color="gray", alpha=0.6)
    axes[0].fill_between(data.index, data["BB_upper"], data["BB_lower"], alpha=0.08, color="gray")
    axes[0].axvline(x=data.index[-1], color=color, linestyle="--", linewidth=1.5, label=f"NOW: {verdict}")
    axes[0].set_title(f"{ticker} — Price / MA / Bollinger Band")
    axes[0].legend(fontsize=8)

    axes[1].plot(data["RSI"], color="purple", linewidth=1.5)
    axes[1].axhline(70, color="red",   linestyle="--", alpha=0.7, label="Overbought (70)")
    axes[1].axhline(30, color="green", linestyle="--", alpha=0.7, label="Oversold (30)")
    axes[1].fill_between(data.index, 30, data["RSI"], where=(data["RSI"] < 30), alpha=0.3, color="green")
    axes[1].fill_between(data.index, 70, data["RSI"], where=(data["RSI"] > 70), alpha=0.3, color="red")
    axes[1].set_title("RSI")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 100)

    axes[2].plot(data["MACD"],        label="MACD",      linewidth=1.5)
    axes[2].plot(data["MACD_signal"], label="Signal",    linewidth=1.5)
    axes[2].bar(data.index, data["MACD"] - data["MACD_signal"],
                color=["green" if x > 0 else "red" for x in data["MACD"] - data["MACD_signal"]],
                alpha=0.5, label="Histogram")
    axes[2].axhline(0, color="gray", linestyle="--")
    axes[2].set_title("MACD")
    axes[2].legend(fontsize=8)

    clrs = ["green" if r > 0 else "red" for r in data["Return_1d"]]
    axes[3].bar(data.index, data["Volume"], color=clrs, alpha=0.7)
    axes[3].plot(data["Volume"].rolling(20).mean(), color="orange", linewidth=1.5, label="20-day avg")
    axes[3].set_title("Volume")
    axes[3].legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    # --- SHAP ---
    X_test     = X.iloc[split:]
    explainer  = shap.TreeExplainer(model)
    sv_all     = explainer.shap_values(X_test)
    sv_today_r = explainer.shap_values(X.iloc[[-1]])

    if isinstance(sv_all, list):
        sv_all     = sv_all[1]
        sv_today   = sv_today_r[1][0]
    else:
        sv_today   = sv_today_r[0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = ["green" if v > 0 else "red" for v in sv_today]
    y_pos  = np.arange(len(FEATURES))
    bars   = axes[0].barh(y_pos, sv_today, color=colors, alpha=0.8)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([FEAT_NAMES[f] for f in FEATURES])
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title(f"{ticker} — Today's SHAP\nUpside prob {ml_prob*100:.1f}%")
    axes[0].set_xlabel("SHAP Value (positive=buy, negative=sell)")
    for bar, val in zip(bars, sv_today):
        axes[0].text(
            val + (0.001 if val >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=9
        )

    mean_shap  = np.abs(sv_all).mean(axis=0)
    sorted_idx = np.argsort(mean_shap)
    axes[1].barh(
        [FEAT_NAMES[FEATURES[i]] for i in sorted_idx],
        mean_shap[sorted_idx], color="steelblue", alpha=0.8
    )
    axes[1].set_title(f"{ticker} — Feature Importance\n(Mean |SHAP|)")
    axes[1].set_xlabel("Mean |SHAP Value|")
    plt.tight_layout()
    plt.show()

    # SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv_all, X_test,
                      feature_names=[FEAT_NAMES[f] for f in FEATURES], show=False)
    plt.title(f"{ticker} — SHAP Summary Plot")
    plt.tight_layout()
    plt.show()

    # Today's SHAP text
    print("[Today's prediction breakdown]")
    for f, v in sorted(zip(FEATURES, sv_today), key=lambda x: abs(x[1]), reverse=True):
        direction = "📈 buy"  if v > 0 else "📉 sell"
        print(f"  {FEAT_NAMES[f]:<30}: {v:+.4f}  {direction}")
    print(f"\n  → {verdict} (upside prob {ml_prob*100:.1f}%)")


# =====================================================
# 3. S&P500 MARKET REBOUND ANALYSIS
# =====================================================
def market_analysis():
    """Estimate S&P500 rebound probability using macro indicators."""
    print(f"\n{'='*50}")
    print(f"  S&P500 Market Rebound Analysis — {date.today()}")
    print(f"{'='*50}")

    spy = yf.download("SPY",  start="2020-01-01", progress=False)
    vix = yf.download("^VIX", start="2020-01-01", progress=False)
    tlt = yf.download("TLT",  start="2020-01-01", progress=False)
    uup = yf.download("UUP",  start="2020-01-01", progress=False)

    for df in [spy, vix, tlt, uup]:
        df.columns = df.columns.get_level_values(0)

    d = pd.DataFrame(index=spy.index)
    d["Close"]        = spy["Close"]
    d["MA20"]         = spy["Close"].rolling(20).mean()
    d["MA60"]         = spy["Close"].rolling(60).mean()
    d["MA200"]        = spy["Close"].rolling(200).mean()
    d["MA_ratio"]     = d["MA20"] / d["MA60"]
    d["MA200_ratio"]  = spy["Close"] / d["MA200"]

    delta             = spy["Close"].diff()
    gain              = delta.clip(lower=0).rolling(14).mean()
    loss              = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"]          = 100 - (100 / (1 + gain / loss))

    bb_mid            = spy["Close"].rolling(20).mean()
    d["BB_position"]  = (spy["Close"] - bb_mid) / (2 * spy["Close"].rolling(20).std())
    d["VIX"]          = vix["Close"]
    d["VIX_high"]     = (vix["Close"] > 30).astype(int)
    d["Return_1d"]    = spy["Close"].pct_change(1)
    d["Return_5d"]    = spy["Close"].pct_change(5)
    d["Return_20d"]   = spy["Close"].pct_change(20)
    d["Volatility"]   = d["Return_1d"].rolling(20).std()
    d["TLT_return"]   = tlt["Close"].pct_change(5)
    d["UUP_return"]   = uup["Close"].pct_change(5)
    d["Future_return"]= spy["Close"].shift(-10) / spy["Close"] - 1
    d["Label"]        = (d["Future_return"] > 0).astype(int)
    d = d.dropna()

    mkt_features = [
        "MA_ratio", "MA200_ratio", "RSI", "BB_position",
        "VIX", "VIX_high", "Return_1d", "Return_5d",
        "Return_20d", "Volatility", "TLT_return", "UUP_return"
    ]
    X     = d[mkt_features]
    y     = d["Label"]
    split = int(len(d) * 0.8)
    model = XGBClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])
    prob  = model.predict_proba(X.iloc[[-1]])[0][1]

    t       = d.iloc[-1]
    price   = float(t["Close"])
    ma20    = float(t["MA20"])
    ma60    = float(t["MA60"])
    ma200   = float(t["MA200"])
    rsi     = float(t["RSI"])
    vix_val = float(t["VIX"])
    ret_5d  = float(t["Return_5d"])
    ret_20d = float(t["Return_20d"])

    sigs = {}
    sigs["MA20/60"]   = "BUY  ✅ Golden" if ma20 > ma60    else "SELL ❌ Dead"
    sigs["MA200"]     = "BUY  ✅ Above 200MA" if price > ma200 else "SELL ❌ Below 200MA"
    if rsi < 30:
        sigs["RSI"]   = f"BUY  ✅ Oversold {rsi:.1f}"
    elif rsi > 70:
        sigs["RSI"]   = f"SELL ❌ Overbought {rsi:.1f}"
    else:
        sigs["RSI"]   = f"NEUT ⚪ Neutral {rsi:.1f}"
    sigs["VIX"]       = f"BUY  ✅ Extreme fear {vix_val:.1f}" if vix_val > 30 else f"NEUT ⚪ Normal {vix_val:.1f}"
    sigs["5d return"] = f"BUY  ✅ {ret_5d*100:.1f}% rebound"  if ret_5d  > 0 else f"SELL ❌ {ret_5d*100:.1f}% falling"
    sigs["20d return"]= f"BUY  ✅ {ret_20d*100:.1f}% up"      if ret_20d > 0 else f"SELL ❌ {ret_20d*100:.1f}% down"
    sigs["ML"]        = f"BUY  ✅ Rebound likely ({prob*100:.1f}%)" if prob > 0.6 else f"SELL ❌ More downside ({prob*100:.1f}%)"

    buy  = sum(1 for v in sigs.values() if "BUY"  in v)
    sell = sum(1 for v in sigs.values() if "SELL" in v)
    neut = sum(1 for v in sigs.values() if "NEUT" in v)

    if   buy >= 5: verdict = "🟢 Strong rebound — consider deploying cash"
    elif buy == 4: verdict = "🟡 Rebound likely — get ready"
    elif buy == 3: verdict = "⚪ Neutral — wait"
    elif sell >= 4:verdict = "🔴 Still downtrend — hold cash"
    else:          verdict = "🟠 Weak sell — observe"

    print(f"\n  SPY: ${price:.2f}  |  VIX: {vix_val:.1f}\n")
    for n, s in sigs.items():
        print(f"  {n:<12}: {s}")
    print(f"\n  Buy: {buy}  Sell: {sell}  Neutral: {neut}")
    print(f"  Verdict: {verdict}")
    print(f"  ML Rebound Prob: {prob*100:.1f}%")


# =====================================================
# 4. BACKTEST
# =====================================================
def backtest(ticker, capital=CAPITAL, fee_rate=FEE_RATE):
    """Compare MA cross vs ML vs Buy&Hold strategies."""
    print(f"\n{'='*55}")
    print(f"  Backtest: {ticker}  |  Start: {START}  |  Fee: {fee_rate*100}%")
    print(f"{'='*55}")

    data         = load_data(ticker)
    model, split = train_model(data)
    X            = data[FEATURES]
    idx          = data.index

    # --- MA strategy ---
    def run_ma(data, cap, fee):
        sig = (data["MA20"] > data["MA60"]).astype(int).diff()
        cash, shares, port = cap, 0, []
        for i, row in data.iterrows():
            p = float(row["Close"]); c = float(sig.loc[i])
            if c == 1 and cash > 0:
                shares = (cash - cash * fee) / p; cash = 0
            elif c == -1 and shares > 0:
                gross = shares * p; cash = gross - gross * fee; shares = 0
            port.append(cash + shares * p)
        return port

    # --- ML strategy ---
    def run_ml(data, cap, fee):
        preds   = model.predict(X)
        cash, shares, port = cap, 0, [cap] * split
        prev = 0
        for i, (_, row) in enumerate(data.iloc[split:].iterrows()):
            p = float(row["Close"]); sig = int(preds[split + i])
            if sig == 1 and prev == 0 and cash > 0:
                shares = (cash - cash * fee) / p; cash = 0
            elif sig == 0 and prev == 1 and shares > 0:
                gross = shares * p; cash = gross - gross * fee; shares = 0
            port.append(cash + shares * p); prev = sig
        return port

    # --- Buy & Hold ---
    def run_bh(data, cap, fee):
        p0     = float(data["Close"].iloc[0])
        shares = (cap - cap * fee) / p0
        return [shares * float(r["Close"]) for _, r in data.iterrows()]

    ma_p = run_ma(data, capital, fee_rate)
    ml_p = run_ml(data, capital, fee_rate)
    bh_p = run_bh(data, capital, fee_rate)

    n    = min(len(ma_p), len(ml_p), len(bh_p))
    ma_p, ml_p, bh_p, idx = ma_p[:n], ml_p[:n], bh_p[:n], idx[:n]

    ma_f, ma_r, ma_mdd, ma_sh = calc_stats(ma_p, capital)
    ml_f, ml_r, ml_mdd, ml_sh = calc_stats(ml_p, capital)
    bh_f, bh_r, bh_mdd, bh_sh = calc_stats(bh_p, capital)

    print(f"\n{'Strategy':<15} {'Final':>10} {'Return':>8} {'MDD':>8} {'Sharpe':>8}")
    print(f"{'-'*55}")
    print(f"{'MA Cross':<15} ${ma_f:>9.2f} {ma_r:>7.1f}% {ma_mdd:>7.1f}% {ma_sh:>8.2f}")
    print(f"{'ML (XGBoost)':<15} ${ml_f:>9.2f} {ml_r:>7.1f}% {ml_mdd:>7.1f}% {ml_sh:>8.2f}")
    print(f"{'Buy & Hold':<15} ${bh_f:>9.2f} {bh_r:>7.1f}% {bh_mdd:>7.1f}% {bh_sh:>8.2f}")
    print(f"{'='*55}")
    best = max([("MA Cross", ma_r), ("ML", ml_r), ("Buy & Hold", bh_r)], key=lambda x: x[1])
    print(f"\n  🏆 Best: {best[0]} ({best[1]:.1f}%)")
    print(f"  MDD: smaller is safer  |  Sharpe: >1 is good, >2 is great")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(idx, ma_p, label=f"MA Cross ({ma_r:.1f}%)",    color="orange", linewidth=2)
    ax1.plot(idx, ml_p, label=f"ML ({ml_r:.1f}%)",          color="blue",   linewidth=2)
    ax1.plot(idx, bh_p, label=f"Buy & Hold ({bh_r:.1f}%)",  color="green",  linewidth=2)
    ax1.axhline(capital, color="gray", linestyle="--", alpha=0.7, label=f"Start (${capital})")
    ax1.set_title(f"{ticker} — Strategy Comparison (fee {fee_rate*100}%)")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(); ax1.grid(alpha=0.3)

    for arr, lbl, col, mdd in [
        (ma_p, f"MA MDD ({ma_mdd:.1f}%)",   "orange", ma_mdd),
        (ml_p, f"ML MDD ({ml_mdd:.1f}%)",   "blue",   ml_mdd),
        (bh_p, f"B&H MDD ({bh_mdd:.1f}%)",  "green",  bh_mdd),
    ]:
        s   = pd.Series(arr)
        dd  = (s - s.cummax()) / s.cummax() * 100
        ax2.plot(idx, dd.values, label=lbl, linewidth=1.5)

    ax2.fill_between(idx,
                     ((pd.Series(ml_p) - pd.Series(ml_p).cummax()) / pd.Series(ml_p).cummax() * 100).values,
                     0, alpha=0.1, color="blue")
    ax2.set_title("Drawdown (%)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# =====================================================
# 5. MASTER RUNNER
# =====================================================
def run(ticker):
    """
    Run full analysis pipeline for a single ticker.
    Includes: signals, SHAP, charts, backtest.
    """
    analyze(ticker)
    backtest(ticker)


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    # Daily monitor (run every morning before market open)
    daily_monitor()

    # Full analysis + backtest for RTX
    run("RTX")

    # Market rebound check (when deciding to deploy remaining $100)
    market_analysis()
