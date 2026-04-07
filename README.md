# quant-trading-system

A personal quantitative trading system that combines classical technical analysis with machine learning to generate buy/sell signals, explain predictions with SHAP values, and backtest strategies against historical data.

Built with Python, XGBoost, and SHAP — designed to run in JupyterLab or as a standalone script.

---

## Features

- **6-indicator signal engine** — MA Cross, RSI, MACD, Bollinger Bands, Volume, and ML combined into a single verdict
- **XGBoost ML model** — trained on historical price features to predict 5-day forward returns
- **SHAP explainability** — visualizes exactly why the model made each prediction
- **Backtest framework** — compares ML strategy vs MA Cross vs Buy & Hold with realistic fee deduction (0.1%)
- **Performance metrics** — final return, Max Drawdown (MDD), and Sharpe Ratio
- **S&P500 macro analysis** — estimates market rebound probability using VIX, bond, and dollar signals
- **Daily monitor** — one-command signal check across your entire watchlist

---

## Quickstart

### 1. Install dependencies

```bash
pip install yfinance xgboost scikit-learn shap matplotlib pandas numpy
```

### 2. Run full analysis

```python
from quant_system import run, daily_monitor, market_analysis, backtest

# Full pipeline: signals + SHAP + charts + backtest
run("RTX")

# Daily watchlist check (run every morning before market open)
daily_monitor()

# Check S&P500 rebound probability before deploying cash
market_analysis()

# Backtest only
backtest("GOOGL")
```

---

## How It Works

### Signal Engine

Each ticker is evaluated across 6 independent indicators. Signals are aggregated into a final verdict.

| Indicator | Logic |
|---|---|
| MA Cross | Golden Cross (MA20 > MA60) → Buy / Dead Cross → Sell |
| RSI | < 30 Oversold → Buy / > 70 Overbought → Sell |
| MACD | MACD line above signal line → Bullish momentum |
| Bollinger Bands | Price below lower band → Oversold bounce signal |
| Volume | Volume ratio > 1.2x 20-day average → Confirms move |
| ML (XGBoost) | Predicts 5-day forward return > 0 → Buy |

**Verdict logic:**
- 4+ Buy signals → 🟢 Strong Buy
- 3 Buy signals → 🟡 Weak Buy
- 4+ Sell signals → 🔴 Strong Sell / Hold Cash
- 3 Sell signals → 🟠 Weak Sell
- Otherwise → ⚪ Neutral

### ML Model

The XGBoost classifier is trained on 80% of historical data and tested on the remaining 20%. Features include:

- `MA_ratio` — MA20 / MA60 ratio
- `RSI` — 14-period relative strength index
- `Volume_ratio` — today's volume vs 20-day average
- `Return_1d` — 1-day price return
- `Return_5d` — 5-day price return
- `Volatility` — 20-day rolling standard deviation of returns

Label: `1` if price is higher 5 trading days later, `0` otherwise.

### SHAP Explainability

After prediction, SHAP (SHapley Additive exPlanations) breaks down exactly which features pushed the model toward buy or sell — and by how much.

```
Today's prediction breakdown (RTX, 2026-04-07):
  Volatility              : +1.4876  📈 buy
  5-Day Return            : +1.1356  📈 buy
  Volume Ratio            : -0.6729  📉 sell
  1-Day Return            : -0.6577  📉 sell
  MA Ratio (MA20/MA60)    : -0.5145  📉 sell
  RSI                     : +0.2144  📈 buy
→ ML Signal: ✅ BUY  (upside prob 80.1%)
```

### Backtest

Strategies are backtested from 2024-01-01 with a 0.1% transaction fee on every trade.

```
RTX Backtest Results (2024-01-01 ~ today)
Capital: $300 | Fee: 0.1%
=======================================================
Strategy        Final $     Return      MDD    Sharpe
-------------------------------------------------------
MA Cross        $ 401.95    34.0%    -17.5%      0.80
ML (XGBoost)    $ 293.95    -2.0%    -11.2%     -0.07
Buy & Hold      $ 606.30   102.1%    -16.2%      1.60
=======================================================
🏆 Best: Buy & Hold (102.1%)
```

> Takeaway: for strongly trending assets like RTX, Buy & Hold wins. ML and MA strategies shine on volatile, mean-reverting tickers.

### S&P500 Macro Analysis

Uses SPY, VIX, TLT (bonds), and UUP (dollar index) to estimate the probability of a market rebound over the next 10 trading days.

```
S&P500 Market Rebound Analysis — 2026-04-07
SPY: $648.57  |  VIX: 26.8

  MA20/60    : SELL ❌ Dead Cross
  MA200      : SELL ❌ Below 200MA
  RSI        : BUY  ✅ Oversold 25.8
  VIX        : NEUT ⚪ Normal 26.8
  5d return  : SELL ❌ -1.8% falling
  20d return : SELL ❌ -5.7% down
  ML         : BUY  ✅ Rebound likely (92.7%)

  Verdict: 🔴 Still downtrend — hold cash
  ML Rebound Prob: 92.7%
```

---

## Project Structure

```
quant-trading-system/
│
├── quant_system.py       # Main system (all functions)
├── README.md             # This file
└── requirements.txt      # Dependencies
```

---

## Configuration

Edit the constants at the top of `quant_system.py`:

```python
START     = "2024-01-01"   # Backtest start date
CAPITAL   = 300            # Starting capital ($)
FEE_RATE  = 0.001          # Transaction fee (0.1%)

WATCHLIST = {
    "RTX":       "Holding ($197.40)",
    "TSLA":      "TSLQ holding - exit on Golden Cross",
    "GOOGL":     "Watchlist",
    "META":      "Watchlist",
}
```

---

## Requirements

```
yfinance
xgboost
scikit-learn
shap
matplotlib
pandas
numpy
```

Or install all at once:

```bash
pip install yfinance xgboost scikit-learn shap matplotlib pandas numpy
```

---

## Disclaimer

This project is for educational purposes only. Nothing in this repository constitutes financial advice. All trading involves risk, and past performance does not guarantee future results. Always do your own research before making any investment decisions.

---

## License

MIT
