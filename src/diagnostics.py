"""Post-estimation diagnostics on standardised residuals.

If the model has captured the volatility dynamics, the standardised residuals
z_t = r_t / sigma_t should be close to i.i.d.: no autocorrelation in z_t, none
in z_t^2, no remaining ARCH effect, and a distribution matching the assumed
innovation law.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from . import config as C
from . import plotting as P


def standardised_residuals(res) -> pd.Series:
    """z_t = (r_t - mu_t) / sigma_t for a fitted arch model."""
    return (res.resid / res.conditional_volatility).dropna()


def residual_tests(fits: dict, label: str, lags: int = 20) -> pd.DataFrame:
    """Ljung-Box, ARCH-LM and normality tests on standardised residuals."""
    rows = []
    for t, by_label in fits.items():
        if label not in by_label:
            continue
        res = by_label[label]
        z = standardised_residuals(res)

        lb_z = acorr_ljungbox(z, lags=[lags], return_df=True)
        lb_z2 = acorr_ljungbox(z**2, lags=[lags], return_df=True)
        arch_stat, arch_p, _, _ = het_arch(z, nlags=lags)
        jb_stat, jb_p = stats.jarque_bera(z)

        rows.append(
            {
                "Index": C.name(t),
                f"LB({lags}) z": lb_z["lb_stat"].iloc[0],
                "p (z)": lb_z["lb_pvalue"].iloc[0],
                f"LB({lags}) z²": lb_z2["lb_stat"].iloc[0],
                "p (z²)": lb_z2["lb_pvalue"].iloc[0],
                f"ARCH-LM({lags})": arch_stat,
                "p (ARCH-LM)": arch_p,
                "Skew(z)": stats.skew(z),
                "Ex. kurtosis(z)": stats.kurtosis(z),
                "JB p": jb_p,
            }
        )
    return pd.DataFrame(rows).set_index("Index")


def residual_test_summary(tests: pd.DataFrame, lags: int = 20) -> pd.DataFrame:
    """Turn the p-values into a plain pass/fail read at the 5% level."""
    out = pd.DataFrame(index=tests.index)
    out["No autocorrelation in z"] = np.where(tests["p (z)"] > 0.05, "pass", "fail")
    out["No autocorrelation in z²"] = np.where(tests["p (z²)"] > 0.05, "pass", "fail")
    out["No remaining ARCH"] = np.where(tests["p (ARCH-LM)"] > 0.05, "pass", "fail")
    out["Residuals normal"] = np.where(tests["JB p"] > 0.05, "pass", "fail")
    return out


def compare_residual_acf(fits: dict, labels: list[str], lags: int = 20) -> pd.DataFrame:
    """Ljung-Box p-value on z² for several specifications side by side."""
    rows = []
    for t, by_label in fits.items():
        row = {"Index": C.name(t)}
        for label in labels:
            if label not in by_label:
                continue
            z = standardised_residuals(by_label[label])
            lb = acorr_ljungbox(z**2, lags=[lags], return_df=True)
            row[label] = lb["lb_pvalue"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows).set_index("Index")


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def plot_residual_diagnostics(fits: dict, label: str, lags: int = 40) -> str:
    """Three-row panel per index: z over time, ACF of z², Q-Q vs assumed law."""
    from statsmodels.tsa.stattools import acf

    tickers = [t for t in C.INDICES if label in fits.get(t, {})]
    n = len(tickers)
    fig, axes = plt.subplots(3, n, figsize=(3.2 * n, 8.4))

    for j, t in enumerate(tickers):
        res = fits[t][label]
        z = standardised_residuals(res)
        col = P.color(t)

        ax = axes[0, j]
        ax.plot(z.index, z.values, color=col, lw=0.4)
        ax.axhline(0, color=C.GRID, lw=0.8)
        ax.set_title(C.name(t), loc="left", fontsize=10)
        # A tick every year is unreadable in a narrow panel.
        ax.xaxis.set_major_locator(mdates.YearLocator(8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        if j == 0:
            ax.set_ylabel("Standardised\nresidual z")

        ax = axes[1, j]
        band = 1.96 / np.sqrt(len(z))
        vals = acf(z**2, nlags=lags, fft=True)[1:]
        ax.bar(range(1, lags + 1), vals, color=col, width=0.85, lw=0)
        ax.axhline(0, color=C.INK_SECONDARY, lw=0.8)
        ax.axhline(band, color=C.INK_MUTED, lw=0.8, ls="--")
        ax.axhline(-band, color=C.INK_MUTED, lw=0.8, ls="--")
        ax.set_ylim(-0.09, 0.09)
        ax.set_xlabel("Lag (days)")
        if j == 0:
            ax.set_ylabel("ACF of z²")

        ax = axes[2, j]
        _qq_against_model(ax, z, res, col)
        ax.set_xlabel("Theoretical quantiles")
        if j == 0:
            ax.set_ylabel("Sample quantiles of z")

    fig.suptitle(
        f"Residual diagnostics — {label}. Volatility clustering is gone from z²; "
        "the tails are what remains to judge.",
        x=0.005,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return P.save(fig, "13_residual_diagnostics.png")


def _qq_against_model(ax, z: pd.Series, res, col: str) -> None:
    """Q-Q plot of z against the innovation distribution the model assumed."""
    n = len(z)
    probs = (np.arange(1, n + 1) - 0.5) / n
    dist_name = res.model.distribution.name.lower()

    if "normal" in dist_name or "gaussian" in dist_name:
        theo = stats.norm.ppf(probs)
        tag = "Normal"
    elif "skew" in dist_name:
        theo = res.model.distribution.ppf(probs, res.params[["nu", "lambda"]].values)
        tag = "Skew-t"
    else:
        nu = float(res.params["nu"])
        theo = res.model.distribution.ppf(probs, np.array([nu]))
        tag = f"Student-t (ν={nu:.1f})"

    theo = np.asarray(theo, dtype=float)
    ax.plot(theo, np.sort(z.values), ls="none", marker="o", ms=1.6, mfc=col, mec=col)
    lo, hi = np.nanmin(theo), np.nanmax(theo)
    ax.plot([lo, hi], [lo, hi], color=C.INK_PRIMARY, lw=1.2, ls="--")
    ax.text(
        0.04, 0.93, tag, transform=ax.transAxes, fontsize=8.5,
        color=C.INK_MUTED, va="top",
    )
