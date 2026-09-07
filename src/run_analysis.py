"""End-to-end pipeline: data -> EDA -> GARCH -> diagnostics -> VaR -> outputs.

Run from the project root:

    python -m src.run_analysis              # full run
    python -m src.run_analysis --skip-var   # skip the slow VaR backtest
    python -m src.run_analysis --refresh    # re-download prices
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from . import config as C
from . import data as D
from . import diagnostics as G
from . import eda as E
from . import models as M
from . import plotting as P
from . import var_backtest as V
from .io_utils import save_table


def main(skip_var: bool = False, refresh: bool = False) -> dict:
    t0 = time.time()
    P.apply_style()
    artefacts: dict[str, list[str]] = {"figures": [], "tables": []}

    def fig(path: str) -> None:
        artefacts["figures"].append(path)
        print(f"    figure -> {path.split('/')[-1]}")

    def tbl(df: pd.DataFrame, name: str, fmt: str = "%.4f") -> None:
        artefacts["tables"].append(save_table(df, name, fmt))
        print(f"    table  -> {name}.csv")

    # ---------------------------------------------------------------- data
    print("[1/6] Loading prices")
    prices = D.load_prices(force=refresh)
    returns = D.log_returns(prices)
    returns, flagged = D.clean_returns(returns)
    if flagged:
        print(f"    dropped implausible observations: {flagged}")
    D.save_processed(prices, returns)
    coverage = D.coverage_table(prices, returns)
    print(coverage.to_string())
    tbl(coverage, "00_data_coverage", "%.1f")

    # ----------------------------------------------------------------- EDA
    print("[2/6] Exploratory analysis")
    summary = E.summary_statistics(returns)
    tbl(summary, "01_summary_statistics")
    tbl(E.stationarity_table(returns), "02_stationarity_adf")
    tbl(E.volatility_clustering_table(returns), "03_volatility_clustering")
    corr = E.correlation_matrix(returns)
    tbl(corr, "04_return_correlation", "%.3f")

    fig(E.plot_normalised_prices(prices))
    fig(E.plot_return_series(returns))
    fig(E.plot_return_distributions(returns))
    fig(E.plot_rolling_volatility(returns))
    fig(E.plot_acf_panel(returns))
    fig(E.plot_correlation_heatmap(corr))

    # -------------------------------------------------------------- models
    print("[3/6] Fitting GARCH specifications")
    fits = M.fit_all(returns)
    for t, by_label in fits.items():
        print(f"    {C.name(t):<16} {len(by_label)} models fitted")

    baseline = C.BASELINE_LABEL
    params_baseline = M.parameter_table(fits, baseline)
    tbl(params_baseline, "05_garch11_normal_parameters")
    tbl(M.parameter_table(fits, "GARCH(1,1)-t"), "06_garch11_t_parameters")
    tbl(M.parameter_table(fits, "GJR-GARCH(1,1)-t"), "07_gjr_garch_t_parameters")
    tbl(M.parameter_table_with_tstats(fits, baseline), "08_garch11_normal_full_estimates")
    tbl(M.parameter_table_with_tstats(fits, "GARCH(1,1)-t"), "09_garch11_t_full_estimates")

    comparison = M.model_comparison(fits)
    tbl(comparison, "10_model_comparison_aic_bic", "%.2f")
    best_bic = M.best_models(comparison, "BIC")
    best_aic = M.best_models(comparison, "AIC")
    best = pd.DataFrame({"Best by AIC": best_aic, "Best by BIC": best_bic})
    tbl(best, "11_preferred_models", "%.0f")
    print(best.to_string())

    regimes = M.regime_table(fits, baseline)
    tbl(regimes, "12_volatility_by_regime", "%.2f")
    print(regimes.round(2).to_string())

    fig(M.plot_conditional_volatility(returns, fits, baseline))
    fig(M.plot_volatility_by_regime(regimes, baseline))
    fig(M.plot_persistence_comparison(params_baseline, baseline))
    fig(M.plot_information_criteria(comparison))
    fig(M.plot_news_impact(fits, "GJR-GARCH(1,1)-t"))

    # Yahoo's EURO STOXX history is short, so re-rank on the common window.
    common = M.common_sample(returns)
    print(f"    common-sample refit from {common.index.min().date()}")
    fits_common = M.fit_all(common, specs=[s for s in C.MODEL_SPECS
                                           if s["label"] in {baseline, "GARCH(1,1)-t"}])
    pers_summary = M.persistence_summary(fits, fits_common, baseline)
    tbl(pers_summary, "13_persistence_full_vs_common")
    print(pers_summary.round(4).to_string())
    fig(M.plot_persistence_full_vs_common(pers_summary, common.index.min()))

    cond_vol = M.conditional_volatility(fits, baseline)
    cond_vol.rename(columns=C.INDICES).to_csv(
        C.DATA_PROCESSED / "conditional_volatility_garch11_normal.csv",
        index_label="date",
    )

    forecasts = M.volatility_forecast(fits, baseline, horizon=60)
    tbl(forecasts, "14_volatility_forecast_60d", "%.3f")
    fig(M.plot_volatility_forecast(fits, forecasts, params_baseline, baseline))

    # ---------------------------------------------------------- diagnostics
    print("[4/6] Residual diagnostics")
    for label, name in [
        (baseline, "15_residual_tests_garch11_normal"),
        ("GARCH(1,1)-t", "16_residual_tests_garch11_t"),
        ("GJR-GARCH(1,1)-t", "17_residual_tests_gjr_t"),
    ]:
        tests = G.residual_tests(fits, label)
        tbl(tests, name)
        if label == baseline:
            tbl(G.residual_test_summary(tests), name + "_verdict", "%.0f")
    tbl(
        G.compare_residual_acf(fits, [s["label"] for s in C.MODEL_SPECS]),
        "18_ljungbox_z2_by_model",
    )
    # Leftover autocorrelation in z is a mean-equation problem, so test the fix.
    t_spec = next(s for s in C.MODEL_SPECS if s["label"] == "GARCH(1,1)-t")
    mean_rob = M.mean_model_robustness(returns, t_spec)
    tbl(mean_rob, "19_mean_model_robustness")
    print(mean_rob.round(4).to_string())
    fig(G.plot_residual_diagnostics(fits, "GARCH(1,1)-t"))

    # ------------------------------------------------------------------ VaR
    if not skip_var:
        print("[5/6] Out-of-sample VaR backtest (this is the slow step)")
        var_specs = [
            s
            for s in C.MODEL_SPECS
            if s["label"] in {"GARCH(1,1)-Normal", "GARCH(1,1)-t", "GJR-GARCH(1,1)-t"}
        ]
        var_frames: dict[tuple[str, str], pd.DataFrame] = {}
        for t in returns.columns:
            r = D.series_for_model(returns, t)
            for spec in var_specs:
                tic = time.time()
                var_frames[(t, spec["label"])] = V.rolling_var_forecast(r, spec)
                print(f"    {C.name(t):<16} {spec['label']:<20} {time.time()-tic:5.1f}s")
            var_frames[(t, "Historical simulation")] = V.historical_simulation_var(r)

        bt = V.backtest_table(var_frames)
        tbl(bt, "20_var_backtest", "%.4f")
        print(
            bt[["Actual breaches", "Expected breaches", "Breach rate", "Verdict"]]
            .round(4)
            .to_string()
        )
        fig(V.plot_var_exceedances(var_frames, "GARCH(1,1)-t", 0.01))
        fig(V.plot_breach_rates(bt))

        for (t, model), frame in var_frames.items():
            if model == "GARCH(1,1)-t":
                frame.to_csv(
                    C.DATA_PROCESSED / f"var_{t.replace('^','')}_garch_t.csv",
                    index_label="date",
                )
    else:
        print("[5/6] Skipping VaR backtest")

    # -------------------------------------------------------------- summary
    print("[6/6] Writing run summary")
    summary_payload = {
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "sample_start": str(returns.index.min().date()),
        "sample_end": str(returns.index.max().date()),
        "indices": {C.name(t): int(returns[t].notna().sum()) for t in returns.columns},
        "baseline_model": baseline,
        "persistence": {
            k: round(float(v), 5) for k, v in params_baseline["alpha+beta"].items()
        },
        "half_life_days": {
            k: round(float(v), 1) for k, v in params_baseline["Half-life (days)"].items()
        },
        "best_by_bic": best_bic.to_dict(),
        "figures": len(artefacts["figures"]),
        "tables": len(artefacts["tables"]),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(C.OUTPUTS / "run_summary.json", "w") as fh:
        json.dump(summary_payload, fh, indent=2)

    print(
        f"\nDone in {summary_payload['runtime_seconds']}s — "
        f"{len(artefacts['figures'])} figures, {len(artefacts['tables'])} tables"
    )
    return {"fits": fits, "returns": returns, "prices": prices, **artefacts}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-var", action="store_true", help="skip the VaR backtest")
    ap.add_argument("--refresh", action="store_true", help="re-download price data")
    args = ap.parse_args()
    main(skip_var=args.skip_var, refresh=args.refresh)
