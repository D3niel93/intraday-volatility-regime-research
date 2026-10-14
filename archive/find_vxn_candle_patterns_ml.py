"""Leakage-controlled ML search for 15-minute VXN candle patterns.

Model selection uses only 2024. Models are frozen and evaluated separately on
2025 and 2026. All features are known at the close of the current bar and all
targets refer to the immediately following 15-minute bar.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    2024: ROOT / "outputs/vxn_multiframe_persistence_causal_2024_2025/events_2024.csv",
    2025: ROOT / "outputs/vxn_multiframe_persistence_causal_2024_2025/events_2025.csv",
    2026: ROOT / "outputs/vxn_multiframe_persistence_frozen_2026/events_2026.csv",
}
OUT = ROOT / "outputs/vxn_candle_patterns_ml"


FEATURES = [
    "bar_fraction", "time_sin", "time_cos", "previous_vxn_close",
    "predicted_range_points", "range_ratio", "body_to_prediction",
    "body_to_range", "upper_wick_to_prediction", "lower_wick_to_prediction",
    "close_location", "direction", "above_prediction", "range_ratio_lag1",
    "range_ratio_lag2", "range_ratio_lag3", "direction_lag1", "direction_lag2",
    "direction_lag3", "above_lag1", "above_lag2", "above_lag3",
    "above_streak", "direction_streak", "rolling4_above_count",
    "rolling4_direction_sum", "rolling4_mean_ratio", "session_above_share",
    "session_net_to_prediction",
]


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def consecutive_true(values: pd.Series) -> pd.Series:
    return values.astype(int).groupby((~values).cumsum()).cumsum()


def consecutive_direction(values: pd.Series) -> pd.Series:
    changed = values.ne(values.shift())
    return values.groupby(changed.cumsum()).cumcount().add(1)


def load_year(year: int) -> pd.DataFrame:
    columns = [
        "local_date", "bucket", "bucket_start_dt", "window", "horizon_minutes",
        "previous_vxn_close", "open", "high", "low", "close",
        "predicted_range_points", "range_ratio", "above_prediction",
        "next_open", "next_high", "next_low", "next_close",
        "next_above_prediction", "next_range_ratio",
    ]
    data = pd.read_csv(FILES[year], usecols=columns)
    data = data[
        data.horizon_minutes.eq(15) & data.window.str.contains("16:00", na=False)
    ].copy()
    data["year"] = year
    data["time"] = pd.to_datetime(data.bucket_start_dt, utc=True)
    data["above_prediction"] = as_bool(data.above_prediction)
    data["next_above_prediction"] = as_bool(data.next_above_prediction)
    data = data.sort_values(["local_date", "bucket"]).reset_index(drop=True)

    candle_range = (data.high - data.low).replace(0, np.nan)
    body = data.close - data.open
    direction = np.sign(body).replace(0, 1).astype(int)
    predicted = data.predicted_range_points.replace(0, np.nan)
    data["bar_fraction"] = data.groupby("local_date").cumcount() / 18.0
    angle = 2 * np.pi * data.bar_fraction
    data["time_sin"] = np.sin(angle)
    data["time_cos"] = np.cos(angle)
    data["body_to_prediction"] = body / predicted
    data["body_to_range"] = body / candle_range
    data["upper_wick_to_prediction"] = (data.high - np.maximum(data.open, data.close)) / predicted
    data["lower_wick_to_prediction"] = (np.minimum(data.open, data.close) - data.low) / predicted
    data["close_location"] = (data.close - data.low) / candle_range
    data["direction"] = direction
    data["above_prediction"] = data.above_prediction.astype(int)

    keys = data.groupby("local_date", group_keys=False)
    for lag in (1, 2, 3):
        data[f"range_ratio_lag{lag}"] = keys.range_ratio.shift(lag)
        data[f"direction_lag{lag}"] = keys.direction.shift(lag)
        data[f"above_lag{lag}"] = keys.above_prediction.shift(lag)
    data["above_streak"] = keys.above_prediction.transform(
        lambda values: consecutive_true(values.astype(bool))
    )
    data["direction_streak"] = keys.direction.transform(consecutive_direction)
    data["rolling4_above_count"] = keys.above_prediction.transform(
        lambda values: values.rolling(4, min_periods=1).sum()
    )
    data["rolling4_direction_sum"] = keys.direction.transform(
        lambda values: values.rolling(4, min_periods=1).sum()
    )
    data["rolling4_mean_ratio"] = keys.range_ratio.transform(
        lambda values: values.rolling(4, min_periods=1).mean()
    )
    bar_number = keys.cumcount().add(1)
    cumulative_above = keys.above_prediction.cumsum()
    data["session_above_share"] = cumulative_above / bar_number
    session_open = keys.open.transform("first")
    data["session_net_to_prediction"] = (data.close - session_open) / predicted

    next_direction = np.sign(data.next_close - data.next_open).replace(0, 1).astype(int)
    data["target_next_expansive"] = data.next_above_prediction.astype(int)
    data["target_next_up"] = next_direction.eq(1).astype(int)
    data["target_next_continuation"] = next_direction.eq(data.direction).astype(int)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data


def model_candidates() -> dict:
    return {
        "logistic": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=.25, max_iter=2000, class_weight="balanced")),
        ]),
        "knn_25": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=25, weights="distance")),
        ]),
        "knn_75": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=75, weights="distance")),
        ]),
        "knn_150": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=150, weights="distance")),
        ]),
        "hist_gb": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=.04, max_iter=150, max_leaf_nodes=9,
                min_samples_leaf=60, l2_regularization=2.0, random_state=20260929,
            )),
        ]),
    }


def metric_row(target: str, model: str, split: str, y: np.ndarray, p: np.ndarray) -> dict:
    pred = p >= .5
    return {
        "target": target, "model": model, "split": split, "n": len(y),
        "base_rate_pct": 100 * y.mean(),
        "auc": roc_auc_score(y, p),
        "accuracy_pct": 100 * accuracy_score(y, pred),
        "balanced_accuracy_pct": 100 * balanced_accuracy_score(y, pred),
        "brier": brier_score_loss(y, p),
    }


def confidence_rows(target: str, model: str, year: int, y: np.ndarray, p: np.ndarray) -> list[dict]:
    rows = []
    for threshold in (.55, .60, .65, .70):
        for side, mask in (
            ("positive", p >= threshold),
            ("negative", p <= 1 - threshold),
        ):
            if not mask.any():
                continue
            event = y[mask] if side == "positive" else 1 - y[mask]
            rows.append({
                "target": target, "model": model, "year": year,
                "confidence": threshold, "prediction_side": side,
                "signals": int(mask.sum()), "coverage_pct": 100 * mask.mean(),
                "hit_rate_pct": 100 * event.mean(),
            })
    return rows


def simple_rule_candidates(data: pd.DataFrame) -> dict[str, pd.Series]:
    rr = data.range_ratio
    body = data.body_to_range.abs()
    close = data.close_location
    return {
        "current_above": data.above_prediction.eq(1),
        "current_below": data.above_prediction.eq(0),
        "range_ratio_ge_1_25": rr.ge(1.25),
        "range_ratio_ge_1_50": rr.ge(1.50),
        "range_ratio_le_0_75": rr.le(.75),
        "above_streak_ge_2": data.above_streak.ge(2),
        "above_streak_ge_3": data.above_streak.ge(3),
        "above_streak_ge_4": data.above_streak.ge(4),
        "four_of_four_above": data.rolling4_above_count.eq(4),
        "zero_of_four_above": data.rolling4_above_count.eq(0),
        "same_direction_ge_2": data.direction_streak.ge(2),
        "same_direction_ge_3": data.direction_streak.ge(3),
        "same_direction_ge_4": data.direction_streak.ge(4),
        "strong_body": body.ge(.60),
        "weak_body": body.le(.25),
        "close_at_high": close.ge(.80),
        "close_at_low": close.le(.20),
        "expansive_strong_body": data.above_prediction.eq(1) & body.ge(.60),
        "expansive_weak_body": data.above_prediction.eq(1) & body.le(.25),
        "expansive_same_dir_ge_3": data.above_prediction.eq(1) & data.direction_streak.ge(3),
        "three_above_same_dir": data.above_streak.ge(3) & data.direction_streak.ge(3),
        "upper_rejection": data.upper_wick_to_prediction.ge(.35) & close.le(.40),
        "lower_rejection": data.lower_wick_to_prediction.ge(.35) & close.ge(.60),
    }


def mine_rules(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    targets = ["target_next_expansive", "target_next_continuation", "target_next_up"]
    rows = []
    names = list(simple_rule_candidates(frames[2024]).keys())
    for target in targets:
        baseline = {year: frame[target].mean() for year, frame in frames.items()}
        for name in names:
            for year, frame in frames.items():
                mask = simple_rule_candidates(frame)[name]
                sample = frame[mask]
                rows.append({
                    "target": target, "rule": name, "year": year,
                    "n": len(sample), "event_rate_pct": 100 * sample[target].mean(),
                    "baseline_pct": 100 * baseline[year],
                    "lift_pp": 100 * (sample[target].mean() - baseline[year]),
                })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {year: load_year(year) for year in (2024, 2025, 2026)}
    targets = ["target_next_expansive", "target_next_continuation", "target_next_up"]
    metrics, confidence, predictions = [], [], []

    dates = np.array(sorted(frames[2024].local_date.unique()))
    cutoff = dates[int(len(dates) * .70)]
    fit24 = frames[2024][frames[2024].local_date.lt(cutoff)]
    val24 = frames[2024][frames[2024].local_date.ge(cutoff)]

    for target in targets:
        candidates = model_candidates()
        validation = []
        for name, model in candidates.items():
            model.fit(fit24[FEATURES], fit24[target])
            p = model.predict_proba(val24[FEATURES])[:, 1]
            row = metric_row(target, name, "2024_validation", val24[target].to_numpy(), p)
            metrics.append(row)
            validation.append(row)
        best_name = max(validation, key=lambda row: row["auc"])["model"]
        best = model_candidates()[best_name]
        best.fit(frames[2024][FEATURES], frames[2024][target])
        for year in (2025, 2026):
            test = frames[year]
            y = test[target].to_numpy()
            p = best.predict_proba(test[FEATURES])[:, 1]
            metrics.append(metric_row(target, best_name, str(year), y, p))
            confidence.extend(confidence_rows(target, best_name, year, y, p))
            predictions.append(pd.DataFrame({
                "local_date": test.local_date, "bucket": test.bucket,
                "year": year, "target": target, "model": best_name,
                "actual": y, "probability": p,
            }))

    pd.DataFrame(metrics).to_csv(OUT / "model_metrics.csv", index=False)
    pd.DataFrame(confidence).to_csv(OUT / "confidence_bands.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "oos_predictions_2025_2026.csv", index=False)
    rules = mine_rules(frames)
    rules.to_csv(OUT / "simple_rule_stability.csv", index=False)

    wide = rules.pivot_table(
        index=["target", "rule"], columns="year", values=["n", "lift_pp", "event_rate_pct"]
    ).reset_index()
    wide.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else col
        for col in wide.columns
    ]
    stable = wide[
        (wide.n_2025 >= 100) & (wide.n_2026 >= 60)
        & (np.sign(wide.lift_pp_2024) == np.sign(wide.lift_pp_2025))
        & (np.sign(wide.lift_pp_2024) == np.sign(wide.lift_pp_2026))
    ].copy()
    stable["min_abs_lift_pp"] = stable[["lift_pp_2024", "lift_pp_2025", "lift_pp_2026"]].abs().min(axis=1)
    stable = stable.sort_values(["target", "min_abs_lift_pp"], ascending=[True, False])
    stable.to_csv(OUT / "stable_rules.csv", index=False)

    print("MODEL METRICS")
    print(pd.DataFrame(metrics).to_string(index=False))
    print("\nSTABLE RULES")
    print(stable.to_string(index=False))


if __name__ == "__main__":
    main()
