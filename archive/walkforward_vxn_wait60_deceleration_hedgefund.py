"""Institutional-style walk-forward audit for the VXN wait60+deceleration rule.

The signal and parameter universe are frozen.  Each expanding fold selects an
interior TP/SL cell by the 25th percentile expectancy of its complete 3x3
neighborhood, then trades the next chronological block.  Only stitched OOS
returns are used for final inference.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/vxn_5m_probability_reversal_sweep_2024_2025/wait60_deceleration_tp10_20_sl10_30_grid/trade_level_grid.parquet"
OUT = ROOT / "outputs/vxn_5m_probability_reversal_sweep_2024_2025/walkforward_hedgefund"
TPS = tuple(range(10, 21, 2))
SLS = tuple(range(10, 31, 2))
INTERIOR_TPS = TPS[1:-1]
INTERIOR_SLS = SLS[1:-1]
POINT_VALUE = 2.0
BOOTSTRAPS = 20_000
SEED = 2601007
BLOCK = 5

FOLDS = (
    ("2025 Q3", "2025-06-30", "2025-07-01", "2025-09-30"),
    ("2025 Q4", "2025-09-30", "2025-10-01", "2025-12-31"),
    ("2026 Q1", "2025-12-31", "2026-01-01", "2026-03-31"),
    ("2026 Q2", "2026-03-31", "2026-04-01", "2026-06-30"),
    ("2026 Jul-Aug", "2026-06-30", "2026-07-01", "2026-08-31"),
)


def basic_metrics(group: pd.DataFrame, pnl_col: str = "net_points") -> dict:
    pnl = group[pnl_col].astype(float)
    curve = pnl.cumsum()
    drawdown = curve - curve.cummax().clip(lower=0)
    positive = pnl[pnl > 0].sum()
    negative = -pnl[pnl < 0].sum()
    dates = pd.to_datetime(group.local_date)
    daily = group.assign(date=dates).groupby("date")[pnl_col].sum()
    calendar = pd.bdate_range(dates.min(), dates.max()) if len(dates) else pd.DatetimeIndex([])
    daily = daily.reindex(calendar, fill_value=0.0)
    std = daily.std(ddof=1)
    downside = daily[daily < 0].std(ddof=1)
    return {
        "trades": len(group), "wins": int(pnl.gt(0).sum()),
        "win_rate_pct": 100 * pnl.gt(0).mean(),
        "net_points": pnl.sum(), "net_usd_1mnq": POINT_VALUE * pnl.sum(),
        "expectancy_points": pnl.mean(),
        "profit_factor": positive / negative if negative else np.inf,
        "max_drawdown_points": drawdown.min(),
        "max_drawdown_usd": POINT_VALUE * drawdown.min(),
        "daily_sharpe": np.sqrt(252) * daily.mean() / std if std > 0 else np.nan,
        "daily_sortino": np.sqrt(252) * daily.mean() / downside if downside > 0 else np.nan,
        "positive_months": int(group.assign(month=dates.dt.to_period("M")).groupby("month")[pnl_col].sum().gt(0).sum()),
        "months": int(dates.dt.to_period("M").nunique()),
    }


def select_plateau(train: pd.DataFrame) -> tuple[int, int, pd.DataFrame]:
    means = train.groupby(["tp", "sl"]).net_points.mean()
    pfs = train.groupby(["tp", "sl"]).net_points.apply(
        lambda x: x[x > 0].sum() / -x[x < 0].sum() if (x < 0).any() else np.inf
    )
    rows = []
    for tp in INTERIOR_TPS:
        for sl in INTERIOR_SLS:
            neighborhood = [
                means.loc[(t, s)] for t in TPS if abs(t - tp) <= 2
                for s in SLS if abs(s - sl) <= 2
            ]
            rows.append({
                "tp": tp, "sl": sl,
                "cell_expectancy": means.loc[(tp, sl)],
                "cell_pf": pfs.loc[(tp, sl)],
                "plateau_q25": np.quantile(neighborhood, .25),
                "plateau_mean": np.mean(neighborhood),
                "plateau_worst": np.min(neighborhood),
                "plateau_dispersion": np.std(neighborhood, ddof=1),
            })
    ranking = pd.DataFrame(rows).sort_values(
        ["plateau_q25", "plateau_mean", "cell_pf"], ascending=False
    ).reset_index(drop=True)
    selected = ranking.iloc[0]
    return int(selected.tp), int(selected.sl), ranking


def walk_forward(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows, oos_parts, rank_parts = [], [], []
    for fold, train_end, test_start, test_end in FOLDS:
        train_mask = data.date.le(pd.Timestamp(train_end))
        test_mask = data.date.between(pd.Timestamp(test_start), pd.Timestamp(test_end))
        train = data[train_mask]
        tp, sl, ranking = select_plateau(train)
        ranking.insert(0, "fold", fold)
        rank_parts.append(ranking)
        test_all = data[test_mask]
        selected = test_all[test_all.tp.eq(tp) & test_all.sl.eq(sl)].copy()
        selected["fold"] = fold
        selected["selected_tp"] = tp
        selected["selected_sl"] = sl
        oos_parts.append(selected)

        # Rank of the chosen training cell among all 66 cells in the next block.
        oos_rank = (
            test_all.groupby(["tp", "sl"]).net_points.mean()
            .rank(method="average", pct=True)
            .loc[(tp, sl)]
        )
        row = {
            "fold": fold, "train_start": train.local_date.min(), "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "train_trades": int(len(train[train.tp.eq(10) & train.sl.eq(10)])),
            "selected_tp": tp, "selected_sl": sl,
            "oos_parameter_percentile": 100 * oos_rank,
        }
        row.update(basic_metrics(selected))
        fold_rows.append(row)
    return pd.DataFrame(fold_rows), pd.concat(oos_parts, ignore_index=True), pd.concat(rank_parts, ignore_index=True)


def fixed_comparator(data: pd.DataFrame, tp: int, sl: int, start: str, end: str) -> pd.DataFrame:
    return data[
        data.tp.eq(tp) & data.sl.eq(sl)
        & data.date.between(pd.Timestamp(start), pd.Timestamp(end))
    ].copy()


def block_bootstrap(pnl: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    n = len(pnl)
    starts = np.arange(n)
    records = np.empty((BOOTSTRAPS, 5), dtype=float)
    for b in range(BOOTSTRAPS):
        pieces = []
        while sum(map(len, pieces)) < n:
            start = int(rng.choice(starts))
            idx = (np.arange(start, start + BLOCK) % n).astype(int)
            pieces.append(pnl[idx])
        sample = np.concatenate(pieces)[:n]
        curve = np.cumsum(sample)
        dd = np.min(curve - np.maximum.accumulate(np.maximum(curve, 0)))
        pos = sample[sample > 0].sum(); neg = -sample[sample < 0].sum()
        records[b] = [sample.sum(), sample.mean(), pos / neg if neg else np.inf, dd, (sample > 0).mean()]
    draws = pd.DataFrame(records, columns=["net_points", "expectancy", "profit_factor", "max_drawdown", "win_rate"])
    summary = pd.DataFrame([{
        "iterations": BOOTSTRAPS, "block_trades": BLOCK,
        "prob_net_positive": draws.net_points.gt(0).mean(),
        "net_points_ci_low": draws.net_points.quantile(.025),
        "net_points_ci_high": draws.net_points.quantile(.975),
        "expectancy_ci_low": draws.expectancy.quantile(.025),
        "expectancy_ci_high": draws.expectancy.quantile(.975),
        "pf_ci_low": draws.profit_factor.replace(np.inf, np.nan).quantile(.025),
        "pf_ci_high": draws.profit_factor.replace(np.inf, np.nan).quantile(.975),
        "max_drawdown_median": draws.max_drawdown.median(),
        "max_drawdown_95pct_worse": draws.max_drawdown.quantile(.05),
        "win_rate_ci_low": 100 * draws.win_rate.quantile(.025),
        "win_rate_ci_high": 100 * draws.win_rate.quantile(.975),
    }])
    return draws, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(SOURCE)
    data["date"] = pd.to_datetime(data.local_date)
    folds, oos, rankings = walk_forward(data)
    start, end = FOLDS[0][2], FOLDS[-1][3]
    comparators = {
        "Walk-forward plateau": oos,
        "Fisso TP10/SL10": fixed_comparator(data, 10, 10, start, end),
        "Fisso TP12/SL24": fixed_comparator(data, 12, 24, start, end),
        "Fisso TP10/SL26": fixed_comparator(data, 10, 26, start, end),
    }
    summary = pd.DataFrame([{"strategy": name, **basic_metrics(group)} for name, group in comparators.items()])

    stress_rows = []
    for cost_usd in (1.0, 2.0, 3.0, 4.0):
        stressed = oos.copy()
        extra_points = (cost_usd - 1.0) / POINT_VALUE
        stressed["stressed_points"] = stressed.net_points - extra_points
        stress_rows.append({"round_trip_cost_usd": cost_usd, **basic_metrics(stressed, "stressed_points")})
    stress = pd.DataFrame(stress_rows)

    draws, bootstrap = block_bootstrap(oos.net_points.to_numpy(float))
    pbo_proxy = pd.DataFrame([{
        "folds": len(folds),
        "selected_below_median_oos_folds": int(folds.oos_parameter_percentile.lt(50).sum()),
        "pbo_proxy_pct": 100 * folds.oos_parameter_percentile.lt(50).mean(),
        "positive_oos_folds": int(folds.net_points.gt(0).sum()),
        "median_oos_parameter_percentile": folds.oos_parameter_percentile.median(),
    }])

    folds.to_csv(OUT / "walkforward_folds.csv", index=False)
    oos.to_csv(OUT / "walkforward_oos_trades.csv", index=False)
    rankings.to_csv(OUT / "training_parameter_rankings.csv", index=False)
    summary.to_csv(OUT / "strategy_comparison.csv", index=False)
    stress.to_csv(OUT / "cost_stress.csv", index=False)
    bootstrap.to_csv(OUT / "block_bootstrap_summary.csv", index=False)
    draws.to_parquet(OUT / "block_bootstrap_draws.parquet", index=False)
    pbo_proxy.to_csv(OUT / "pbo_proxy.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    colors = ("#e1812c", "#777777", "#2f6f9f", "#55a868")
    for (name, group), color in zip(comparators.items(), colors):
        group = group.sort_values("entry_ts")
        axes[0, 0].step(pd.to_datetime(group.entry_ts), group.net_points.mul(2).cumsum(),
                        where="post", label=name, color=color, linewidth=1.6)
    axes[0, 0].axhline(0, color="black", linewidth=.8)
    axes[0, 0].set_title("Equity esclusivamente OOS walk-forward")
    axes[0, 0].set_ylabel("P&L ($), 1 MNQ")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=.2)

    axes[0, 1].bar(folds.fold, folds.net_usd_1mnq, color=np.where(folds.net_points >= 0, "#55a868", "#d95f5f"))
    axes[0, 1].axhline(0, color="black", linewidth=.8)
    axes[0, 1].set_title("P&L OOS per fold")
    axes[0, 1].set_ylabel("P&L ($)"); axes[0, 1].tick_params(axis="x", rotation=20)
    axes[0, 1].grid(axis="y", alpha=.2)

    x = np.arange(len(folds))
    axes[1, 0].plot(x, folds.selected_tp, marker="o", label="TP")
    axes[1, 0].plot(x, folds.selected_sl, marker="o", label="SL")
    axes[1, 0].set_xticks(x, folds.fold, rotation=20)
    axes[1, 0].set_title("Parametri scelti usando soltanto il passato")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=.2)

    axes[1, 1].hist(draws.net_points.mul(2), bins=50, color="#7187a3", alpha=.85)
    axes[1, 1].axvline(0, color="black", linewidth=.9)
    axes[1, 1].axvline(2 * oos.net_points.sum(), color="#e1812c", linewidth=1.5, label="Osservato")
    axes[1, 1].set_title(f"Bootstrap a blocchi ({BOOTSTRAPS:,} iterazioni)")
    axes[1, 1].set_xlabel("P&L su 60 trade ($)"); axes[1, 1].legend()
    fig.suptitle("VXN 5m wait60 + decelerazione — audit walk-forward")
    fig.savefig(OUT / "walkforward_hedgefund_dashboard.png", dpi=190)
    plt.close(fig)

    print("FOLDS")
    print(folds.to_string(index=False))
    print("\nSTRATEGIES")
    print(summary.to_string(index=False))
    print("\nCOST STRESS")
    print(stress.to_string(index=False))
    print("\nBOOTSTRAP")
    print(bootstrap.to_string(index=False))
    print("\nPBO PROXY")
    print(pbo_proxy.to_string(index=False))


if __name__ == "__main__":
    main()
