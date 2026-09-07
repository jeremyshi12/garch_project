"""GARCH estimation, model comparison and volatility persistence.

Fitting convention: a constant mean and returns already scaled by 100, so the
estimated omega/alpha/beta are on a percent-squared variance scale.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model

from . import config as C
from . import plotting as P


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
def fit_one(returns: pd.Series, spec: dict, mean: str = "Constant", **mean_kwargs):
    """Fit a single volatility specification to one return series."""
    kwargs = {k: v for k, v in spec.items() if k != "label"}
    kwargs.update(mean_kwargs)
    kwargs.setdefault("mean", mean)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(returns, **kwargs)
        res = am.fit(disp="off", show_warning=False)
    return res


def fit_all(
    returns: pd.DataFrame, specs: list[dict] | None = None
) -> dict[str, dict[str, object]]:
    """Fit every specification to every index.

    Returns {ticker: {label: ARCHModelResult}}.
    """
    specs = specs or C.MODEL_SPECS
    fits: dict[str, dict[str, object]] = {}
    for t in returns.columns:
        r = returns[t].dropna()
        fits[t] = {}
        for spec in specs:
            try:
                fits[t][spec["label"]] = fit_one(r, spec)
            except Exception as exc:  # a spec that will not converge is reported, not fatal
                print(f"  ! {C.name(t)} / {spec['label']} failed: {exc}")
    return fits


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def _p(res, key: str, default: float = 0.0) -> float:
    """Parameter lookup that tolerates a missing term (e.g. no gamma in GARCH)."""
    return float(res.params[key]) if key in res.params.index else default


def persistence(res, label: str) -> float:
    """Volatility persistence for a fitted model.

    GARCH(1,1):     alpha + beta
    GJR-GARCH(1,1): alpha + gamma/2 + beta -- gamma is halved because the
                    leverage term only fires on negative shocks, which happen
                    half the time under a symmetric innovation distribution.
    EGARCH(1,1):    beta, the AR coefficient on log variance.
    """
    if "EGARCH" in label:
        return _p(res, "beta[1]")
    alpha = _p(res, "alpha[1]")
    beta = _p(res, "beta[1]")
    gamma = _p(res, "gamma[1]")
    return alpha + gamma / 2.0 + beta


def half_life(pers: float) -> float:
    """Days for a variance shock to decay halfway back to the long-run level."""
    if not (0 < pers < 1):
        return np.inf
    return float(np.log(0.5) / np.log(pers))


def long_run_vol(res, label: str) -> float:
    """Annualised unconditional volatility implied by the fitted parameters."""
    pers = persistence(res, label)
    if "EGARCH" in label:
        if not 0 < pers < 1:
            return np.nan
        log_var = _p(res, "omega") / (1 - pers)
        return float(np.sqrt(np.exp(log_var)) * np.sqrt(C.TRADING_DAYS))
    if not 0 < pers < 1:
        return np.nan
    var = _p(res, "omega") / (1 - pers)
    return float(np.sqrt(var) * np.sqrt(C.TRADING_DAYS))


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def parameter_table(fits: dict, label: str) -> pd.DataFrame:
    """omega / alpha / beta / alpha+beta for one specification, all indices.

    This is the headline table the project asks for.
    """
    rows = []
    for t, by_label in fits.items():
        if label not in by_label:
            continue
        res = by_label[label]
        pers = persistence(res, label)
        row = {
            "Index": C.name(t),
            "mu": _p(res, "mu"),
            "omega": _p(res, "omega"),
            "alpha": _p(res, "alpha[1]"),
            "beta": _p(res, "beta[1]"),
            "alpha+beta": pers,
            "Half-life (days)": half_life(pers),
            "Long-run ann. vol (%)": long_run_vol(res, label),
            "Log-lik": float(res.loglikelihood),
            "AIC": float(res.aic),
            "BIC": float(res.bic),
        }
        if "gamma[1]" in res.params.index:
            row["gamma"] = _p(res, "gamma[1]")
        if "nu" in res.params.index:
            row["nu (t dof)"] = _p(res, "nu")
        rows.append(row)
    return pd.DataFrame(rows).set_index("Index")


def parameter_table_with_tstats(fits: dict, label: str) -> pd.DataFrame:
    """Estimates with standard errors and t-statistics, long format."""
    frames = []
    for t, by_label in fits.items():
        if label not in by_label:
            continue
        res = by_label[label]
        df = pd.DataFrame(
            {
                "Estimate": res.params,
                "Std. error": res.std_err,
                "t-stat": res.tvalues,
                "p-value": res.pvalues,
            }
        )
        df.insert(0, "Index", C.name(t))
        df.index.name = "Parameter"
        frames.append(df.reset_index())
    out = pd.concat(frames, ignore_index=True)
    return out.set_index(["Index", "Parameter"])


def model_comparison(fits: dict) -> pd.DataFrame:
    """AIC / BIC / log-likelihood for every (index, specification) pair."""
    rows = []
    for t, by_label in fits.items():
        for label, res in by_label.items():
            pers = persistence(res, label)
            rows.append(
                {
                    "Index": C.name(t),
                    "Model": label,
                    "Params": len(res.params),
                    "Log-lik": float(res.loglikelihood),
                    "AIC": float(res.aic),
                    "BIC": float(res.bic),
                    "Persistence": pers,
                    "Half-life (days)": half_life(pers),
                }
            )
    df = pd.DataFrame(rows)
    # Rank within each index so "best" is unambiguous.
    df["AIC rank"] = df.groupby("Index")["AIC"].rank().astype(int)
    df["BIC rank"] = df.groupby("Index")["BIC"].rank().astype(int)
    return df.set_index(["Index", "Model"]).sort_index()


def best_models(comparison: pd.DataFrame, criterion: str = "BIC") -> pd.Series:
    """The preferred specification per index under AIC or BIC."""
    idx = comparison.reset_index().groupby("Index")[criterion].idxmin()
    return comparison.reset_index().loc[idx].set_index("Index")["Model"]


def mean_model_robustness(
    returns: pd.DataFrame, spec: dict, lags: int = 20
) -> pd.DataFrame:
    """Constant mean vs AR(1) mean, judged on leftover autocorrelation in z.

    The Ljung-Box test on standardised residuals is a test of the *mean*
    equation, not the variance equation. If it rejects, the fix is a richer
    mean model, not a richer GARCH.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    rows = []
    for t in returns.columns:
        r = returns[t].dropna()
        row = {"Index": C.name(t)}
        for mean_label, mean_kw in [("Constant", {"mean": "Constant"}),
                                    ("AR(1)", {"mean": "AR", "lags": 1})]:
            res = fit_one(r, spec, **mean_kw)
            z = (res.resid / res.conditional_volatility).dropna()
            lb = acorr_ljungbox(z, lags=[lags], return_df=True)
            row[f"LB({lags}) p — {mean_label}"] = lb["lb_pvalue"].iloc[0]
            row[f"BIC — {mean_label}"] = float(res.bic)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Index")


def common_sample(returns: pd.DataFrame) -> pd.DataFrame:
    """Truncate the panel to the window every index actually covers.

    Yahoo only serves EURO STOXX 50 from 2007, so a full-sample persistence
    ranking would compare a 26-year US sample against a 19-year European one.
    Refitting on the common window is the honest like-for-like check.
    """
    latest_start = max(returns[c].dropna().index.min() for c in returns.columns)
    return returns.loc[latest_start:]


def persistence_summary(
    fits_full: dict, fits_common: dict, label: str
) -> pd.DataFrame:
    """Persistence and half-life on the full vs the common sample, side by side."""
    rows = []
    for t in fits_full:
        if label not in fits_full.get(t, {}) or label not in fits_common.get(t, {}):
            continue
        pf = persistence(fits_full[t][label], label)
        pc = persistence(fits_common[t][label], label)
        rows.append(
            {
                "Index": C.name(t),
                "alpha+beta (full)": pf,
                "Half-life (full)": half_life(pf),
                "alpha+beta (common)": pc,
                "Half-life (common)": half_life(pc),
                "Difference": pc - pf,
            }
        )
    return pd.DataFrame(rows).set_index("Index")


def plot_persistence_full_vs_common(summary: pd.DataFrame, common_start) -> str:
    """Does the persistence ranking survive putting every market on one window?"""
    tbl = summary.sort_values("alpha+beta (full)")
    y = np.arange(len(tbl))
    fig, ax = plt.subplots(figsize=(9, 4.4))

    for i, idx_name in enumerate(tbl.index):
        ticker = next(t for t in C.INDICES if C.name(t) == idx_name)
        col = P.color(ticker)
        ax.barh(i + 0.19, tbl["alpha+beta (full)"].iloc[i], height=0.34,
                color=col, zorder=3)
        ax.barh(i - 0.19, tbl["alpha+beta (common)"].iloc[i], height=0.34,
                color=col, zorder=3, hatch="////", edgecolor=C.SURFACE, lw=0.0)
        ax.text(tbl["alpha+beta (full)"].iloc[i] - 0.0012, i + 0.19,
                f"{tbl['alpha+beta (full)'].iloc[i]:.4f}", va="center", ha="right",
                fontsize=8.5, color="#ffffff", fontweight="bold", zorder=5)
        # White-on-hatch is unreadable, so the common-sample value sits outside
        # its bar in ink instead.
        ax.text(tbl["alpha+beta (common)"].iloc[i] + 0.0015, i - 0.19,
                f"{tbl['alpha+beta (common)'].iloc[i]:.4f}", va="center", ha="left",
                fontsize=8.5, color=C.INK_SECONDARY, zorder=5)

    ax.set_yticks(y, tbl.index)
    ax.set_xlim(0.9, 1.0)
    ax.set_xlabel("alpha + beta")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.75, len(tbl) - 0.25)

    # Identity is the entity colour; the two samples are told apart by texture.
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=C.INK_MUTED, label="Full available sample"),
            Patch(facecolor=C.INK_MUTED, hatch="////", edgecolor=C.SURFACE,
                  label=f"Common sample (from {pd.Timestamp(common_start).date()})"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.34),
        ncols=2,
    )
    P.titled(
        ax,
        "Persistence is robust to the sample window",
        "Same four markets, refit on the window all of them cover.",
    )
    fig.tight_layout()
    return P.save(fig, "16_persistence_full_vs_common.png")


def conditional_volatility(fits: dict, label: str) -> pd.DataFrame:
    """Annualised conditional volatility series for one specification."""
    out = {}
    for t, by_label in fits.items():
        if label not in by_label:
            continue
        res = by_label[label]
        out[t] = res.conditional_volatility * np.sqrt(C.TRADING_DAYS)
    return pd.DataFrame(out).sort_index()


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def plot_conditional_volatility(
    returns: pd.DataFrame, fits: dict, label: str
) -> str:
    """Per-index conditional volatility with the return series behind it."""
    tickers = [t for t in returns.columns if label in fits.get(t, {})]
    n = len(tickers)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.3 * n), sharex=True)

    for ax, t in zip(np.atleast_1d(axes), tickers):
        res = fits[t][label]
        r = returns[t].dropna()
        cv = res.conditional_volatility * np.sqrt(C.TRADING_DAYS)
        ax.plot(r.index, r.abs().values, color=C.GRID, lw=0.4, zorder=1)
        ax.plot(cv.index, cv.values, color=P.color(t), lw=1.1, zorder=2)
        ax.set_ylabel("Ann. vol (%)")
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
        P.shade_crises(ax)

    axes[0].set_title(
        f"Conditional volatility from {label} (annualised), over |daily return|",
        loc="left",
    )
    fig.align_ylabels(axes)
    return P.save(fig, "07_conditional_volatility.png")


def regime_table(fits: dict, label: str) -> pd.DataFrame:
    """Mean conditional volatility inside each crisis window vs calm periods.

    Answers the project's "do crisis periods show clear spikes?" question with
    a number instead of an eyeball read of a chart.
    """
    cv = conditional_volatility(fits, label)
    in_any_crisis = pd.Series(False, index=cv.index)
    cols: dict[str, pd.Series] = {}

    for start, end, tag in C.CRISIS_PERIODS:
        mask = (cv.index >= pd.Timestamp(start)) & (cv.index <= pd.Timestamp(end))
        in_any_crisis |= mask
        cols[tag] = cv[mask].mean()

    out = pd.DataFrame({"Calm periods": cv[~in_any_crisis.values].mean(), **cols})
    out.index = [C.name(t) for t in out.index]
    out.insert(
        len(out.columns),
        "Peak / calm ratio",
        out.drop(columns="Calm periods").max(axis=1) / out["Calm periods"],
    )
    return out


def plot_volatility_by_regime(regimes: pd.DataFrame, label: str) -> str:
    """Grouped bars: conditional volatility in calm markets vs each crisis."""
    cols = [c for c in regimes.columns if c != "Peak / calm ratio"]
    order = [C.name(t) for t in C.INDICES if C.name(t) in regimes.index]
    fig, ax = plt.subplots(figsize=(11, 4.8))

    x = np.arange(len(cols))
    width = 0.8 / len(order)
    for k, idx_name in enumerate(order):
        ticker = next(t for t in C.INDICES if C.name(t) == idx_name)
        ax.bar(
            x + k * width - 0.4 + width / 2,
            regimes.loc[idx_name, cols].values,
            width=width * 0.86,
            color=P.color(ticker),
            label=idx_name,
            zorder=3,
        )

    ax.set_xticks(x, [c.replace(" / ", "/\n").replace(" crisis", "\ncrisis") for c in cols])
    ax.set_ylabel("Mean ann. conditional volatility (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=4)
    P.titled(
        ax,
        f"Conditional volatility by regime — {label}",
        "Every market roughly doubles its calm-period volatility in a crisis, but "
        "which crisis hurts most is regional: Europe in 2011, the US in 2020.",
    )
    fig.tight_layout()
    return P.save(fig, "08_volatility_by_regime.png")


def plot_persistence_comparison(param_table: pd.DataFrame, label: str) -> str:
    """The project's headline chart: alpha+beta and its half-life, per market."""
    tbl = param_table.sort_values("alpha+beta")
    tickers = [t for t, nm in C.INDICES.items() if nm in tbl.index]
    colors = [P.color(next(t for t in tickers if C.name(t) == nm)) for nm in tbl.index]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    y = np.arange(len(tbl))
    ax.barh(y, tbl["alpha+beta"], color=colors, height=0.55, zorder=3)
    ax.set_yticks(y, tbl.index)
    ax.set_xlim(0.9, 1.0)
    ax.axvline(1.0, color=C.STATUS["critical"], lw=1.2, ls="--", zorder=4)
    ax.text(
        0.9995,
        len(tbl) - 0.35,
        "unit persistence",
        rotation=90,
        ha="right",
        va="top",
        fontsize=8,
        color=C.STATUS["critical"],
    )
    for i, v in enumerate(tbl["alpha+beta"]):
        ax.text(v - 0.0012, i, f"{v:.4f}", va="center", ha="right",
                fontsize=9, color="#ffffff", fontweight="bold", zorder=5)
    ax.set_xlabel("alpha + beta")
    ax.set_title("Volatility persistence", loc="left")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    hl = tbl["Half-life (days)"]
    ax.barh(y, hl, color=colors, height=0.55, zorder=3)
    ax.set_yticks(y, tbl.index)
    for i, v in enumerate(hl):
        ax.text(v - max(hl) * 0.015, i, f"{v:.0f}d", va="center", ha="right",
                fontsize=9, color="#ffffff", fontweight="bold", zorder=5)
    ax.set_xlabel("Half-life of a variance shock (trading days)")
    ax.set_title("What that persistence means", loc="left")
    ax.grid(axis="y", visible=False)

    fig.suptitle(
        f"Volatility persistence across markets — {label}",
        x=0.005,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return P.save(fig, "09_persistence_comparison.png")


def plot_information_criteria(comparison: pd.DataFrame) -> str:
    """BIC relative to each index's best model — lower is better, 0 is the winner."""
    df = comparison.reset_index()
    df["dBIC"] = df.groupby("Index")["BIC"].transform(lambda s: s - s.min())
    df["dAIC"] = df.groupby("Index")["AIC"].transform(lambda s: s - s.min())

    models = [s["label"] for s in C.MODEL_SPECS]
    order = [C.name(t) for t in C.INDICES]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)

    for ax, crit in zip(axes, ["dAIC", "dBIC"]):
        width = 0.8 / len(order)
        x = np.arange(len(models))
        for k, idx_name in enumerate(order):
            sub = df[df["Index"] == idx_name].set_index("Model").reindex(models)
            ticker = next(t for t in C.INDICES if C.name(t) == idx_name)
            ax.bar(
                x + k * width - 0.4 + width / 2,
                sub[crit].values,
                width=width * 0.88,
                color=P.color(ticker),
                label=idx_name,
                zorder=3,
            )
        ax.set_xticks(x, models, rotation=20, ha="right")
        ax.set_title(
            f"Δ{crit[1:]} vs each index's best model", loc="left"
        )
        ax.set_ylabel("Δ (lower is better)" if crit == "dAIC" else "")
        ax.grid(axis="x", visible=False)

    axes[0].legend(loc="upper right", ncols=2)
    fig.suptitle(
        "Model selection: normal innovations are decisively rejected in every market",
        x=0.005,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return P.save(fig, "10_information_criteria.png")


def plot_news_impact(fits: dict, label: str) -> str:
    """News impact curve: how a shock today maps into variance tomorrow."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    shocks = np.linspace(-4, 4, 401)
    ends: list[tuple[float, float, str, str]] = []

    for t, by_label in fits.items():
        if label not in by_label:
            continue
        res = by_label[label]
        sigma = float(np.sqrt(np.nanmean(res.conditional_volatility**2)))
        eps = shocks * sigma
        omega, alpha = _p(res, "omega"), _p(res, "alpha[1]")
        beta, gamma = _p(res, "beta[1]"), _p(res, "gamma[1]")
        var_next = omega + alpha * eps**2 + gamma * eps**2 * (eps < 0) + beta * sigma**2
        vol_next = np.sqrt(var_next * C.TRADING_DAYS)
        ax.plot(shocks, vol_next, color=P.color(t), lw=1.6)
        ends.append((shocks[-1], float(vol_next[-1]), C.name(t), P.color(t)))

    ax.axvline(0, color=C.GRID, lw=1.0)
    P.label_line_ends(ax, ends)
    P.titled(
        ax,
        f"News impact curve — {label}",
        "Next-day volatility as a function of today's shock, in standard deviations. "
        "The left arm rises steeply and the right arm is nearly flat: only bad news "
        "moves volatility.",
        wrap=72,
    )
    ax.set_xlabel("Standardised shock today (σ)")
    ax.set_ylabel("Implied next-day ann. volatility (%)")
    # Direct labels carry identity here; a legend as well would be redundant.
    fig.subplots_adjust(right=0.78)
    return P.save(fig, "11_news_impact_curve.png")


def volatility_forecast(fits: dict, label: str, horizon: int = 60) -> pd.DataFrame:
    """Analytic multi-step variance forecast, converted to annualised vol."""
    out = {}
    for t, by_label in fits.items():
        if label not in by_label:
            continue
        res = by_label[label]
        try:
            f = res.forecast(horizon=horizon, reindex=False, method="analytic")
        except (ValueError, NotImplementedError):
            f = res.forecast(horizon=horizon, reindex=False, method="simulation")
        var = f.variance.iloc[-1].values
        out[C.name(t)] = np.sqrt(var * C.TRADING_DAYS)
    return pd.DataFrame(out, index=pd.RangeIndex(1, horizon + 1, name="Horizon (days)"))


def plot_volatility_forecast(
    fits: dict, forecasts: pd.DataFrame, param_table: pd.DataFrame, label: str
) -> str:
    """Forecast term structure converging to each market's long-run level."""
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ends = []
    for idx_name in forecasts.columns:
        ticker = next(t for t in C.INDICES if C.name(t) == idx_name)
        s = forecasts[idx_name]
        ax.plot(s.index, s.values, color=P.color(ticker), lw=1.7)
        ends.append((s.index[-1], float(s.iloc[-1]), idx_name, P.color(ticker)))
        if idx_name in param_table.index:
            lr = param_table.loc[idx_name, "Long-run ann. vol (%)"]
            ax.axhline(lr, color=P.color(ticker), lw=0.9, ls=":", alpha=0.75)

    P.titled(
        ax,
        f"Volatility term structure forecast — {label}",
        "Solid: forecast from the last observed day. Dotted: fitted long-run level. "
        "The gap closes at a speed set by alpha+beta.",
        wrap=76,
    )
    ax.set_xlabel("Forecast horizon (trading days)")
    ax.set_ylabel("Annualised volatility (%)")
    # Direct labels only — S&P and FTSE converge, so they are nudged apart.
    P.label_line_ends(ax, ends)
    fig.subplots_adjust(right=0.78)
    return P.save(fig, "12_volatility_forecast.png")
