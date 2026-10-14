"""Five-minute VXN range/reversal study across contraction probabilities.

The logistic classifier is fitted on causally constructed 2024 five-minute
events and evaluated on frozen 2025 events.  Reversal outcomes use the first
raw MNQ tick touching the forecast band and preserve true tick ordering.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_vxn_logistic_range_model import FEATURES, model
from analyze_vxn_contractive_band_reversals_2024_2025 import (
    build_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/vxn_multiframe_persistence_causal_2024_2025"
OUT = ROOT / "outputs/vxn_5m_probability_reversal_sweep_2024_2025"
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().map({"true": True, "false": False}).astype(bool)


def consecutive_direction(values: pd.Series) -> pd.Series:
    groups = values.ne(values.shift()).cumsum()
    return values.groupby(groups).cumcount().add(1) * values


def feature_engineer(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    data = frame.copy().sort_values(["local_date", "bucket"]).reset_index(drop=True)
    data["year"] = year
    data["time"] = pd.to_datetime(data.bucket_start_dt, utc=True)
    data["above_prediction"] = as_bool(data.above_prediction).astype(int)
    data["next_above_prediction"] = as_bool(data.next_above_prediction).astype(int)
    candle_range = (data.high - data.low).replace(0, np.nan)
    body = data.close - data.open
    predicted = data.predicted_range_points.replace(0, np.nan)
    # The 16:00-21:00 Italian window contains 60 five-minute bars.
    data["bar_fraction"] = data.bucket / 60.0
    angle = 2 * np.pi * data.bar_fraction
    data["time_sin"] = np.sin(angle)
    data["time_cos"] = np.cos(angle)
    data["body_to_prediction"] = body / predicted
    data["body_to_range"] = body / candle_range
    data["upper_wick_to_prediction"] = (data.high - np.maximum(data.open, data.close)) / predicted
    data["lower_wick_to_prediction"] = (np.minimum(data.open, data.close) - data.low) / predicted
    data["close_location"] = (data.close - data.low) / candle_range
    data["direction"] = np.sign(body).replace(0, 1).astype(int)
    keys = data.groupby("local_date", group_keys=False)
    for lag in (1, 2, 3):
        data[f"range_ratio_lag{lag}"] = keys.range_ratio.shift(lag)
        data[f"direction_lag{lag}"] = keys.direction.shift(lag)
        data[f"above_lag{lag}"] = keys.above_prediction.shift(lag)
    data["above_streak"] = keys.above_prediction.transform(
        lambda x: x.groupby((x == 0).cumsum()).cumsum()
    )
    data["direction_streak"] = keys.direction.transform(consecutive_direction)
    data["rolling4_above_count"] = keys.above_prediction.transform(
        lambda x: x.rolling(4, min_periods=1).sum()
    )
    data["rolling4_direction_sum"] = keys.direction.transform(
        lambda x: x.rolling(4, min_periods=1).sum()
    )
    data["rolling4_mean_ratio"] = keys.range_ratio.transform(
        lambda x: x.rolling(4, min_periods=1).mean()
    )
    count = keys.cumcount().add(1)
    data["session_above_share"] = keys.above_prediction.cumsum() / count
    session_open = keys.open.transform("first")
    data["session_net_to_prediction"] = (data.close - session_open) / predicted
    data["target_next_expansive"] = data.next_above_prediction.astype(int)

    # Forecast distance for the next five-minute bucket, known at signal close.
    next_bucket = keys.bucket.shift(-1)
    next_multiplier = keys.vxn_multiplier_2024.shift(-1)
    contiguous = next_bucket.eq(data.bucket + 1)
    data["next_forecast_points"] = (
        data.close * data.raw_sigma_pct * next_multiplier / 100.0
    )
    data["target_start"] = data.time + pd.Timedelta(minutes=5)
    data["target_end"] = data.target_start + pd.Timedelta(minutes=5)
    data["upper_band"] = data.close + data.next_forecast_points
    data["lower_band"] = data.close - data.next_forecast_points
    data["probability_expansive"] = np.nan
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data[contiguous & data.next_forecast_points.notna()].copy()


def load_events() -> dict[int, pd.DataFrame]:
    result = {}
    for year in (2024, 2025):
        raw = pd.read_csv(SOURCE / f"events_{year}.csv")
        raw = raw[
            raw.horizon_minutes.eq(5)
            & raw.window.astype(str).str.contains("16:00", regex=False)
        ].copy()
        result[year] = feature_engineer(raw, year)
    return result


def barrier_accuracy(group: pd.DataFrame, points: int) -> tuple[int, float]:
    values = group[group[f"barrier{points}_60m"].ne("unresolved")][f"barrier{points}_60m"]
    return len(values), 100 * values.eq("tp").mean() if len(values) else np.nan


def summarize(events: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in (2024, 2025):
        annual_events = events[events.year.eq(year)]
        annual_ledger = ledger[ledger.year.eq(year)]
        for threshold in THRESHOLDS:
            max_p = 1 - threshold
            eligible = annual_events[annual_events.probability_expansive.le(max_p)]
            touched = annual_ledger[annual_ledger.probability_expansive.le(max_p)]
            row = {
                "year": year,
                "probability_contractive_threshold": threshold,
                "eligible_signals": len(eligible),
                "actual_contractive_accuracy_pct": 100 * (1 - eligible.target_next_expansive).mean(),
                "touches": len(touched),
                "touch_rate_pct": 100 * len(touched) / len(eligible) if len(eligible) else np.nan,
            }
            for label, group in (
                ("all", touched),
                ("upper_short", touched[touched.side.eq("upper")]),
                ("lower_long", touched[touched.side.eq("lower")]),
            ):
                row[f"{label}_touches"] = len(group)
                for points in (10, 20, 30):
                    n, accuracy = barrier_accuracy(group, points)
                    row[f"{label}_resolved_{points}"] = n
                    row[f"{label}_reversal_{points}_pct"] = accuracy
            rows.append(row)
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {2024: "#7187a3", 2025: "#e1812c"}
    for year in (2024, 2025):
        data = summary[summary.year.eq(year)]
        x = 100 * data.probability_contractive_threshold
        axes[0, 0].plot(x, data.actual_contractive_accuracy_pct, marker="o", color=colors[year], label=str(year))
        axes[0, 1].plot(x, data.eligible_signals, marker="o", color=colors[year], label=str(year))
        axes[1, 0].plot(x, data.upper_short_reversal_10_pct, marker="o", color=colors[year], label=str(year))
        axes[1, 1].plot(x, data.upper_short_reversal_20_pct, marker="o", color=colors[year], label=str(year))
        for ax, ycol in ((axes[1, 0], "upper_short_resolved_10"), (axes[1, 1], "upper_short_resolved_20")):
            for xx, yy, n in zip(x, data[ycol.replace("resolved", "reversal").replace("_10", "_10_pct").replace("_20", "_20_pct")], data[ycol]):
                if np.isfinite(yy):
                    ax.annotate(f"n={int(n)}", (xx, yy), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    axes[0, 0].set_title("Accuratezza reale: barra 5m contrattiva")
    axes[0, 0].set_ylabel("Accuratezza (%)")
    axes[0, 1].set_title("Numero di segnali disponibili")
    axes[0, 1].set_ylabel("Segnali")
    axes[1, 0].set_title("Tocco banda superiore: reversal 10 pt")
    axes[1, 0].set_ylabel("TP prima dello stop simmetrico (%)")
    axes[1, 1].set_title("Tocco banda superiore: reversal 20 pt")
    for ax in axes.flat:
        ax.set_xlabel("Soglia probabilità contrattiva (%)")
        ax.grid(alpha=.2)
        ax.legend()
    axes[0, 0].set_ylim(45, 100)
    axes[1, 0].axhline(50, color="black", linestyle="--", linewidth=.8)
    axes[1, 1].axhline(50, color="black", linestyle="--", linewidth=.8)
    fig.suptitle("MNQ 5 minuti — soglia di confidenza VXN e qualità del reversal")
    fig.savefig(OUT / "five_minute_probability_sweep.png", dpi=190)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    yearly = load_events()
    classifier = model().fit(yearly[2024][FEATURES], yearly[2024].target_next_expansive)
    parts = []
    for year, frame in yearly.items():
        frame = frame.copy()
        frame["probability_expansive"] = classifier.predict_proba(frame[FEATURES])[:, 1]
        parts.append(frame)
    events = pd.concat(parts, ignore_index=True)
    candidates = events[events.probability_expansive.le(.50)].copy()
    ledger = build_ledger(candidates)
    summary = summarize(events, ledger)
    events.to_parquet(OUT / "all_5m_predictions.parquet", index=False)
    ledger.to_parquet(OUT / "tick_touch_ledger.parquet", index=False)
    summary.to_csv(OUT / "probability_threshold_summary.csv", index=False)
    plot(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
