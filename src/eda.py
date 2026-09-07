"""Exploratory analysis: summary statistics, distribution and volatility charts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from . import config as C
from . import plotting as P


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def summary_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-index descriptive statistics for daily log returns (in percent)."""
    rows = []
    for t in returns.columns:
        r = returns[t].dropna()
        ann_vol = r.std() * np.sqrt(C.TRADING_DAYS)
        jb_stat, jb_p = stats.jarque_bera(r)
        rows.append(
            {
                "Index": C.name(t),
                "Obs": len(r),
                "Mean (%/day)": r.mean(),
                "Ann. return (%)": r.mean() * C.TRADING_DAYS,
                "Std (%/day)": r.std(),
                "Ann. vol (%)": ann_vol,
                "Skew": stats.skew(r),
                "Excess kurtosis": stats.kurtosis(r),  # already excess
                "Min (%)": r.min(),
                "Max (%)": r.max(),
                "JB stat": jb_stat,
                "JB p": jb_p,
            }
        )
    return pd.DataFrame(rows).set_index("Index")


def stationarity_table(returns: pd.DataFrame) -> pd.DataFrame:
    """Augmented Dickey-Fuller test on returns: are they stationary?"""
    rows = []
    for t in returns.columns:
        r = returns[t].dropna()
        stat, pval, lags, nobs, crit, _ = adfuller(r, autolag="AIC")
        rows.append(
            {
                "Index": C.name(t),
                "ADF stat": stat,
                "p-value": pval,
                "Lags used": lags,
                "Crit. 1%": crit["1%"],
                "Crit. 5%": crit["5%"],
                "Stationary at 5%": "Yes" if pval < 0.05 else "No",
            }
        )
    return pd.DataFrame(rows).set_index("Index")


def volatility_clustering_table(returns: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    """Ljung-Box on returns vs squared returns.

    The signature of volatility clustering: returns themselves show little
    autocorrelation, but their squares are strongly autocorrelated.
    """
    rows = []
    for t in returns.columns:
        r = returns[t].dropna()
        lb_r = acorr_ljungbox(r, lags=[lags], return_df=True)
        lb_r2 = acorr_ljungbox(r**2, lags=[lags], return_df=True)
        rows.append(
            {
                "Index": C.name(t),
                f"LB({lags}) returns": lb_r["lb_stat"].iloc[0],
                "p (returns)": lb_r["lb_pvalue"].iloc[0],
                f"LB({lags}) squared": lb_r2["lb_stat"].iloc[0],
                "p (squared)": lb_r2["lb_pvalue"].iloc[0],
                "ACF |r| lag1": r.abs().autocorr(1),
            }
        )
    return pd.DataFrame(rows).set_index("Index")


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise return correlation on the common trading calendar."""
    common = returns.dropna()
    corr = common.corr()
    corr.index = [C.name(t) for t in corr.index]
    corr.columns = [C.name(t) for t in corr.columns]
    return corr


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def plot_normalised_prices(prices: pd.DataFrame) -> str:
    """Index levels rebased to 100 at the first common date."""
    fig, ax = plt.subplots(figsize=(10, 5))
    base = prices.dropna().iloc[0]
    norm = prices.divide(base).multiply(100)

    ends = []
    for t in prices.columns:
        s = norm[t].dropna()
        ax.plot(s.index, s.values, color=P.color(t), lw=1.5)
        ends.append((s.index[-1], float(s.iloc[-1]), C.name(t), P.color(t)))

    P.shade_crises(ax)
    base_date = norm.dropna().index[0].date()
    P.titled(
        ax,
        "Index levels, rebased to 100",
        f"Price indices only, dividends excluded. Base is {base_date}, the first date "
        "all four cover — earlier values are shown against that same base.",
        wrap=88,
    )
    ax.set_ylabel("Level (rebased, log scale)")
    ax.set_yscale("log")
    # Direct labels only; a legend as well would be redundant.
    P.label_line_ends(ax, ends)
    ax.margins(x=0.02)
    fig.subplots_adjust(right=0.84)
    return P.save(fig, "01_prices_rebased.png")


def plot_return_series(returns: pd.DataFrame) -> str:
    """Small multiples of daily returns — the visual case for GARCH."""
    n = len(returns.columns)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.1 * n), sharex=True)
    for ax, t in zip(np.atleast_1d(axes), returns.columns):
        r = returns[t].dropna()
        ax.plot(r.index, r.values, color=P.color(t), lw=0.45)
        ax.set_ylabel("%")
        ax.text(
            0.005,
            0.93,
            C.name(t),
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            color=C.INK_PRIMARY,
            va="top",
        )
        ax.axhline(0, color=C.GRID, lw=0.8)
    axes[0].set_title(
        "Daily log returns: quiet stretches and violent stretches cluster together",
        loc="left",
    )
    fig.align_ylabels(axes)
    return P.save(fig, "02_return_series.png")


def plot_return_distributions(returns: pd.DataFrame) -> str:
    """Histogram vs fitted normal, plus a normal Q-Q plot, per index."""
    n = len(returns.columns)
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 6.4))

    for j, t in enumerate(returns.columns):
        r = returns[t].dropna()
        col = P.color(t)

        ax = axes[0, j]
        ax.hist(r, bins=140, density=True, color=col, alpha=0.75, lw=0)
        grid = np.linspace(r.min(), r.max(), 500)
        ax.plot(
            grid,
            stats.norm.pdf(grid, r.mean(), r.std()),
            color=C.INK_PRIMARY,
            lw=1.4,
            ls="--",
            label="Normal fit",
        )
        ax.set_yscale("log")
        ax.set_xlim(-8, 8)
        ax.set_title(C.name(t), loc="left", fontsize=10)
        if j == 0:
            ax.set_ylabel("Density (log scale)")
            ax.legend(loc="lower center", fontsize=8)

        ax = axes[1, j]
        stats.probplot(r, dist="norm", plot=ax)
        ax.get_lines()[0].set(marker="o", ms=1.6, mfc=col, mec=col, ls="none")
        ax.get_lines()[1].set(color=C.INK_PRIMARY, lw=1.2, ls="--")
        ax.set_title("")
        ax.set_xlabel("Normal quantiles")
        ax.set_ylabel("Sample quantiles" if j == 0 else "")

    fig.suptitle(
        "Fat tails everywhere: log density is far above the normal in both tails, "
        "and the Q-Q plot bends away at each end",
        x=0.005,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return P.save(fig, "03_return_distributions.png")


def plot_rolling_volatility(returns: pd.DataFrame, window: int = 21) -> str:
    """Annualised rolling standard deviation — the model-free volatility view.

    Small multiples rather than one overlay: four volatility paths on a single
    axis is unreadable spaghetti. Each panel colours its own market and keeps
    the other three in grey behind it, so both the level and the cross-market
    comparison stay legible.
    """
    roll = returns.rolling(window, min_periods=window).std() * np.sqrt(C.TRADING_DAYS)
    tickers = list(returns.columns)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True, sharey=True)

    for ax, t in zip(axes.ravel(), tickers):
        for other in tickers:
            if other == t:
                continue
            s = roll[other].dropna()
            ax.plot(s.index, s.values, color=C.GRID, lw=0.5, zorder=1)
        s = roll[t].dropna()
        ax.plot(s.index, s.values, color=P.color(t), lw=0.8, zorder=3)
        P.shade_crises(ax)
        ax.set_title(C.name(t), loc="left", fontsize=10)
        ax.margins(x=0.01)

    for ax in axes[:, 0]:
        ax.set_ylabel("Annualised volatility (%)")

    fig.suptitle(
        f"{window}-day rolling volatility, annualised — grey lines are the other "
        "three markets",
        x=0.005,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.text(
        0.005,
        0.925,
        "Shaded: crisis windows. The spikes line up across regions; their height "
        "and decay do not.",
        fontsize=9,
        color=C.INK_MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    return P.save(fig, "04_rolling_volatility.png")


def plot_acf_panel(returns: pd.DataFrame, lags: int = 40) -> str:
    """ACF of returns vs squared returns: no memory in the level, lots in the square."""
    from statsmodels.tsa.stattools import acf

    n = len(returns.columns)
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 5.6), sharey="row", sharex=True)

    for j, t in enumerate(returns.columns):
        r = returns[t].dropna()
        col = P.color(t)
        band = 1.96 / np.sqrt(len(r))

        for i, (series, tag) in enumerate([(r, "Returns"), (r**2, "Squared returns")]):
            ax = axes[i, j]
            vals = acf(series, nlags=lags, fft=True)[1:]
            ax.bar(range(1, lags + 1), vals, color=col, width=0.85, lw=0)
            ax.axhline(0, color=C.INK_SECONDARY, lw=0.8)
            ax.axhline(band, color=C.INK_MUTED, lw=0.8, ls="--")
            ax.axhline(-band, color=C.INK_MUTED, lw=0.8, ls="--")
            if i == 0:
                ax.set_title(C.name(t), loc="left", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{tag}\nautocorrelation")
            if i == 1:
                ax.set_xlabel("Lag (days)")

    fig.suptitle(
        "Returns are close to unpredictable; their squares are strongly persistent "
        "— exactly what GARCH is built to model",
        x=0.005,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return P.save(fig, "05_acf_returns_vs_squared.png")


def plot_correlation_heatmap(corr: pd.DataFrame) -> str:
    """Sequential single-hue heatmap of the return correlation matrix."""
    from matplotlib.colors import LinearSegmentedColormap

    # Single-hue blue ramp, light -> dark (sequential encoding of magnitude).
    ramp = LinearSegmentedColormap.from_list(
        "blues", ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
    )
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(corr.values, cmap=ramp, vmin=0, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    ax.grid(False)

    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=9.5,
                color="#ffffff" if v > 0.55 else C.INK_PRIMARY,
            )

    P.titled(ax, "Daily return correlation", "Common trading days only")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    return P.save(fig, "06_return_correlation.png")
