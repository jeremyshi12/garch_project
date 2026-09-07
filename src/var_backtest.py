"""Out-of-sample Value-at-Risk forecasting and backtesting.

Design: an expanding estimation window, refit every `refit_every` trading days.
Between refits the parameters are held fixed but the variance recursion keeps
running on *realised* returns, so every VaR number is a genuine one-step-ahead
forecast that uses no future information.

Backtests: Kupiec unconditional coverage, Christoffersen independence, and the
joint conditional-coverage test.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

from . import config as C
from . import plotting as P


# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------
def _dist_quantile(res_or_model, alpha: float) -> float:
    """Standardised innovation quantile at `alpha` for the fitted distribution."""
    dist = res_or_model.model.distribution
    name = dist.name.lower()
    if "normal" in name or "gaussian" in name:
        return float(stats.norm.ppf(alpha))
    if "skew" in name:
        shape = res_or_model.params[["nu", "lambda"]].values
    else:
        shape = np.array([float(res_or_model.params["nu"])])
    return float(np.asarray(dist.ppf(alpha, shape)).ravel()[0])


def rolling_var_forecast(
    returns: pd.Series,
    spec: dict,
    oos_start: str = C.VAR_OOS_START,
    levels: tuple[float, ...] = C.VAR_LEVELS,
    refit_every: int = C.VAR_REFIT_EVERY,
    mean: str = "Constant",
) -> pd.DataFrame:
    """One-step-ahead VaR for each day in the out-of-sample window.

    Returns a frame indexed by date with the realised return, the forecast
    conditional mean and volatility, and one VaR column per level.
    """
    returns = returns.dropna()
    dates = returns.index
    start_pos = int(dates.searchsorted(pd.Timestamp(oos_start)))
    if start_pos < 500:
        raise ValueError("Need at least 500 in-sample observations before oos_start")

    records = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for block_start in range(start_pos, len(returns), refit_every):
            # Fit on everything strictly before the block.
            train = returns.iloc[:block_start]
            kwargs = {k: v for k, v in spec.items() if k != "label"}
            res = arch_model(train, mean=mean, **kwargs).fit(
                disp="off", show_warning=False
            )

            # Re-run the recursion over the full series with those parameters,
            # so the variance entering each forecast reflects realised returns.
            fixed = arch_model(returns, mean=mean, **kwargs).fix(res.params)
            fc = fixed.forecast(horizon=1, reindex=True, start=block_start - 1)
            sigma = np.sqrt(fc.variance["h.1"])
            mu = fc.mean["h.1"]

            quantiles = {lv: _dist_quantile(res, lv) for lv in levels}
            block_end = min(block_start + refit_every, len(returns))
            for pos in range(block_start, block_end):
                # The forecast made on day pos-1 applies to day pos.
                prev = dates[pos - 1]
                rec = {
                    "date": dates[pos],
                    "return": float(returns.iloc[pos]),
                    "mu": float(mu.loc[prev]),
                    "sigma": float(sigma.loc[prev]),
                }
                for lv in levels:
                    rec[f"VaR_{lv:.0%}"] = rec["mu"] + rec["sigma"] * quantiles[lv]
                records.append(rec)

    return pd.DataFrame(records).set_index("date")


def historical_simulation_var(
    returns: pd.Series,
    oos_start: str = C.VAR_OOS_START,
    levels: tuple[float, ...] = C.VAR_LEVELS,
    window: int = 500,
) -> pd.DataFrame:
    """Benchmark: rolling empirical quantile, no volatility model at all."""
    returns = returns.dropna()
    out = {"return": returns}
    for lv in levels:
        out[f"VaR_{lv:.0%}"] = returns.shift(1).rolling(window).quantile(lv)
    df = pd.DataFrame(out).dropna()
    return df.loc[pd.Timestamp(oos_start):]


# --------------------------------------------------------------------------
# Backtests
# --------------------------------------------------------------------------
def kupiec_pof(violations: np.ndarray, alpha: float) -> tuple[float, float]:
    """Unconditional coverage: is the violation *rate* right? chi2(1)."""
    n = len(violations)
    x = int(violations.sum())
    if x == 0:
        stat = -2 * (n * np.log(1 - alpha))
        return float(stat), float(1 - stats.chi2.cdf(stat, 1))
    pi = x / n
    ll_null = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    ll_alt = (n - x) * np.log(1 - pi) + x * np.log(pi)
    stat = -2 * (ll_null - ll_alt)
    return float(stat), float(1 - stats.chi2.cdf(stat, 1))


def christoffersen_independence(violations: np.ndarray) -> tuple[float, float]:
    """Independence: are violations clustered? chi2(1).

    Clustered breaches are the dangerous failure mode — a model can have the
    right average hit rate and still blow through the limit five days running.
    """
    v = violations.astype(int)
    n00 = int(((v[:-1] == 0) & (v[1:] == 0)).sum())
    n01 = int(((v[:-1] == 0) & (v[1:] == 1)).sum())
    n10 = int(((v[:-1] == 1) & (v[1:] == 0)).sum())
    n11 = int(((v[:-1] == 1) & (v[1:] == 1)).sum())

    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return np.nan, np.nan

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    if pi in (0.0, 1.0) or pi01 in (0.0,) or pi11 in (0.0, 1.0):
        return np.nan, np.nan

    ll_null = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll_alt = (
        n00 * np.log(1 - pi01)
        + n01 * np.log(pi01)
        + n10 * np.log(1 - pi11)
        + n11 * np.log(pi11)
    )
    stat = -2 * (ll_null - ll_alt)
    return float(stat), float(1 - stats.chi2.cdf(stat, 1))


def backtest(var_frame: pd.DataFrame, alpha: float) -> dict:
    """Full backtest report for one series at one VaR level."""
    col = f"VaR_{alpha:.0%}"
    df = var_frame[["return", col]].dropna()
    viol = (df["return"] < df[col]).values
    n, x = len(df), int(viol.sum())

    uc_stat, uc_p = kupiec_pof(viol, alpha)
    ind_stat, ind_p = christoffersen_independence(viol)
    cc_stat = uc_stat + ind_stat if np.isfinite(ind_stat) else np.nan
    cc_p = float(1 - stats.chi2.cdf(cc_stat, 2)) if np.isfinite(cc_stat) else np.nan

    exceed = (df["return"] - df[col])[viol]
    return {
        "Obs": n,
        "Expected breaches": alpha * n,
        "Actual breaches": x,
        "Breach rate": x / n,
        "Target rate": alpha,
        "Kupiec stat": uc_stat,
        "Kupiec p": uc_p,
        "Independence stat": ind_stat,
        "Independence p": ind_p,
        "Cond. coverage stat": cc_stat,
        "Cond. coverage p": cc_p,
        "Avg breach size (%)": float(exceed.mean()) if x else 0.0,
        "Worst breach (%)": float(exceed.min()) if x else 0.0,
        "Mean VaR (%)": float(df[col].mean()),
    }


def backtest_table(
    var_frames: dict[tuple[str, str], pd.DataFrame],
    levels: tuple[float, ...] = C.VAR_LEVELS,
) -> pd.DataFrame:
    """Backtest every (index, model) frame at every level."""
    rows = []
    for (ticker, model), frame in var_frames.items():
        for lv in levels:
            if f"VaR_{lv:.0%}" not in frame:
                continue
            rec = {"Index": C.name(ticker), "Model": model, "Level": f"{lv:.0%}"}
            rec.update(backtest(frame, lv))
            rows.append(rec)
    df = pd.DataFrame(rows)
    df["Verdict"] = np.where(
        df["Cond. coverage p"].fillna(df["Kupiec p"]) > 0.05, "pass", "fail"
    )
    return df.set_index(["Index", "Level", "Model"]).sort_index()


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def plot_var_exceedances(
    var_frames: dict[tuple[str, str], pd.DataFrame], model: str, alpha: float = 0.01
) -> str:
    """Returns against the VaR line, with breaches marked."""
    col = f"VaR_{alpha:.0%}"
    tickers = [t for (t, m) in var_frames if m == model]
    tickers = [t for t in C.INDICES if t in tickers]
    n = len(tickers)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.3 * n), sharex=True)

    for ax, t in zip(np.atleast_1d(axes), tickers):
        df = var_frames[(t, model)]
        viol = df["return"] < df[col]
        ax.plot(df.index, df["return"], color=C.GRID, lw=0.5, zorder=1)
        ax.plot(df.index, df[col], color=P.color(t), lw=1.1, zorder=2)
        ax.scatter(
            df.index[viol],
            df["return"][viol],
            s=14,
            color=C.STATUS["critical"],
            zorder=3,
            label=f"Breach ({int(viol.sum())})",
        )
        ax.set_ylabel("Daily return (%)")
        ax.text(
            0.005, 0.06, C.name(t), transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", color=C.INK_PRIMARY, va="bottom",
        )
        ax.legend(loc="lower right", fontsize=8)

    axes[0].set_title(
        f"{alpha:.0%} one-day VaR vs realised returns — {model} "
        f"(out-of-sample from {C.VAR_OOS_START[:4]})",
        loc="left",
    )
    fig.align_ylabels(axes)
    return P.save(fig, f"14_var_exceedances_{int(alpha*100)}pct.png")


def plot_breach_rates(bt: pd.DataFrame) -> str:
    """Observed breach rate vs target, per model and level."""
    df = bt.reset_index()
    levels = df["Level"].unique()
    fig, axes = plt.subplots(1, len(levels), figsize=(6.0 * len(levels), 4.4))
    axes = np.atleast_1d(axes)

    for ax, lv in zip(axes, levels):
        sub = df[df["Level"] == lv]
        models = list(dict.fromkeys(sub["Model"]))
        order = [C.name(t) for t in C.INDICES if C.name(t) in set(sub["Index"])]
        x = np.arange(len(models))
        width = 0.8 / max(len(order), 1)

        for k, idx_name in enumerate(order):
            s = sub[sub["Index"] == idx_name].set_index("Model").reindex(models)
            ticker = next(t for t in C.INDICES if C.name(t) == idx_name)
            ax.bar(
                x + k * width - 0.4 + width / 2,
                s["Breach rate"].values * 100,
                width=width * 0.88,
                color=P.color(ticker),
                label=idx_name,
                zorder=3,
            )
        target = float(lv.strip("%"))
        ax.axhline(target, color=C.STATUS["critical"], lw=1.3, ls="--", zorder=4)
        # Reserve a strip to the right of the bars so the label never lands on one.
        ax.set_xlim(-0.6, len(models) - 0.1)
        ax.annotate(
            f"target {lv}", xy=(len(models) - 0.55, target), xytext=(2, 4),
            textcoords="offset points", va="bottom", ha="left",
            fontsize=8.5, color=C.STATUS["critical"],
        )
        ax.set_xticks(x, models, rotation=18, ha="right")
        ax.set_ylabel("Observed breach rate (%)")
        ax.set_title(f"{lv} VaR", loc="left")
        ax.grid(axis="x", visible=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper right", ncols=4,
        bbox_to_anchor=(0.995, 0.995), frameon=False,
    )
    fig.suptitle(
        "VaR backtest: how often did losses exceed the limit?",
        x=0.005, ha="left", fontsize=12.5, fontweight="bold",
    )
    fig.text(
        0.005, 0.915,
        "Every model breaches the 1% limit too often — the tails are fatter than "
        "any of them assume.",
        fontsize=9, color=C.INK_MUTED, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return P.save(fig, "15_var_breach_rates.png")
