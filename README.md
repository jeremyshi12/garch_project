# GARCH Models for Global Equity Indices

Estimating GARCH-family volatility models for four major equity indices and comparing
**volatility persistence** across markets.

| Index | Ticker | Region | Sample |
|---|---|---|---|
| S&P 500 | `^GSPC` | United States | 2000–2026 |
| EURO STOXX 50 | `^STOXX50E` | Euro area | 2007–2026 |
| Nikkei 225 | `^N225` | Japan | 2000–2026 |
| FTSE 100 | `^FTSE` | United Kingdom | 2000–2026 |

## Headline findings

- **Volatility is highly persistent everywhere**: α + β runs 0.976–0.981, so a variance
  shock takes 28–36 trading days to decay halfway.
- **The S&P 500 is the most persistent** market on every specification, and the ranking
  survives refitting all four indices on a common sample window.
- **Persistence and level rank differently.** The Nikkei is the most volatile index
  (23.5% annualised) but the quickest to calm down; the S&P is among the calmest yet
  holds a shock the longest.
- **Student-t innovations beat normal decisively**, but **GJR-GARCH wins on both AIC and
  BIC in all four markets** — the leverage effect matters more than the tail assumption.
  In three of four indices the symmetric ARCH term collapses to zero: only *bad* news
  moves volatility.
- **Out of sample, no model gets the 1% tail right.** GARCH fixes breach *clustering*
  (independence p ≈ 0.2–0.7) where historical simulation fails badly (p ≈ 0.000), but
  every model breaches the 1% limit 1.2–2.3× too often.

Full write-up: [`reports/GARCH_report.md`](reports/GARCH_report.md).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Reproduce every figure and table (~50 seconds, downloads data on first run):

```bash
python -m src.run_analysis
```

Useful flags:

```bash
python -m src.run_analysis --skip-var   # skip the slow VaR backtest (~4s total)
python -m src.run_analysis --refresh    # re-download prices from Yahoo Finance
```

Or work through the narrative version in
[`notebooks/garch_analysis.ipynb`](notebooks/garch_analysis.ipynb) — open the project
folder in VS Code and select the `.venv` kernel.

## Project layout

```
garch_project/
├── data/
│   ├── raw/            # cached Yahoo Finance downloads, one CSV per index
│   └── processed/      # cleaned price/return panels, conditional volatility, VaR series
├── notebooks/
│   └── garch_analysis.ipynb
├── src/
│   ├── config.py       # tickers, date range, model specs, palette
│   ├── data.py         # download, cache, log returns, cleaning
│   ├── eda.py          # summary statistics and exploratory charts
│   ├── models.py       # GARCH fitting, persistence, comparison, forecasts
│   ├── diagnostics.py  # standardised-residual tests and plots
│   ├── var_backtest.py # out-of-sample VaR forecasting and backtests
│   ├── plotting.py     # shared matplotlib styling
│   ├── io_utils.py     # table export (CSV + Markdown)
│   └── run_analysis.py # end-to-end pipeline
├── outputs/
│   ├── figures/        # 16 PNGs
│   ├── tables/         # 22 tables, each as .csv and .md
│   └── run_summary.json
├── reports/
│   └── GARCH_report.md
├── requirements.txt
└── README.md
```

## Method notes

**Log returns × 100.** GARCH is fitted by numerical maximum likelihood. Raw daily returns
are ~0.01, which pushes ω to ~1e-6 and makes the optimiser unstable; scaling to percent
puts every parameter near order 1.

**Each index keeps its own trading calendar.** Tokyo, London, New York and the euro area
have different holidays. Forward-filling across a holiday would invent a zero return and
distort exactly the clustering being measured, so returns are computed per index on that
index's own observed days. The panel is outer-joined for charts only.

**Persistence.** α + β for GARCH; α + γ/2 + β for GJR (γ is halved because the leverage
term only fires on negative shocks, which occur half the time under a symmetric
innovation distribution); β alone for EGARCH, where the recursion is in log-variance.
Half-life is ln(0.5)/ln(persistence).

**VaR backtest.** Expanding estimation window, refit every 10 trading days, one-step-ahead
forecasts only. Between refits the parameters are frozen but the variance recursion keeps
running on realised returns, so no forecast uses future information.

## Known limitations

1. Yahoo serves EURO STOXX 50 only from 2007, so it has a shorter history than the other
   three. `src/models.py:common_sample` refits everything on the shared window as a
   robustness check; the persistence ranking is unchanged.
2. Markets close at different times, so a "same day" shock is not the same information
   set across indices. Harmless for the univariate models here, but it would need care in
   a multivariate extension.
3. Returns are unhedged and in local currency.
4. Parameters are assumed constant over 26 years spanning the GFC, COVID and the 2022
   rates shock.

## Data source

Yahoo Finance via [`yfinance`](https://pypi.org/project/yfinance/). Index price data,
dividends excluded. Downloads are cached in `data/raw/`, so the project runs offline
after the first pull.
