"""Full chronology audit and rebuild of the 15-minute VXN logistic range model."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from find_vxn_candle_patterns_ml import FEATURES, FILES, as_bool, consecutive_direction
from tick_integrity import load_ticks_chronological


ROOT = Path(__file__).resolve().parents[1]
TICK_ROOT = ROOT / "tick_backtest_mnq/cache/tick_parquet"
OUT = ROOT / "outputs/vxn_logistic_range_chronology_audit"


def source_events(year: int) -> pd.DataFrame:
    columns = [
        "local_date", "bucket", "bucket_start_dt", "window", "horizon_minutes",
        "previous_vxn_close", "open", "high", "low", "close",
        "predicted_range_points", "range_ratio", "above_prediction",
        "next_open", "next_high", "next_low", "next_close", "next_above_prediction",
    ]
    frame = pd.read_csv(FILES[year], usecols=columns)
    frame = frame[frame.horizon_minutes.eq(15) & frame.window.str.contains("16:00", na=False)].copy()
    frame["above_prediction"] = as_bool(frame.above_prediction)
    frame["next_above_prediction"] = as_bool(frame.next_above_prediction)
    return frame.sort_values(["local_date", "bucket"]).reset_index(drop=True)


def raw_bar(ts_ns: np.ndarray, prices: np.ndarray, start: pd.Timestamp) -> tuple[int, int, int, int]:
    if start.tzinfo is not None:
        start = start.tz_convert("UTC").tz_localize(None)
    left = int(np.searchsorted(ts_ns, start.value, side="left"))
    right = int(np.searchsorted(ts_ns, (start + pd.Timedelta(minutes=15)).value, side="left"))
    sample = prices[left:right]
    if len(sample) == 0:
        raise RuntimeError(f"No raw ticks for 15-minute bar starting {start}")
    return int(sample[0]), int(sample.max()), int(sample.min()), int(sample[-1])


def audit_and_correct(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = source_events(year)
    records = []
    audit_rows = []
    for date, day in frame.groupby("local_date", sort=True):
        path = TICK_ROOT / f"calendar_date={date}" / "data_0.parquet"
        ticks, integrity = load_ticks_chronological(path, ["ts", "price_ticks"])
        ts_ns = ticks.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        prices = ticks.price_ticks.to_numpy(np.int64)
        day_mismatches = {name: 0 for name in ("open", "high", "low", "close", "next_open", "next_high", "next_low", "next_close")}
        for row in day.itertuples(index=True):
            start = pd.Timestamp(row.bucket_start_dt)
            current = raw_bar(ts_ns, prices, start)
            following = raw_bar(ts_ns, prices, start + pd.Timedelta(minutes=15))
            raw_values = {
                "open": current[0] / 4, "high": current[1] / 4,
                "low": current[2] / 4, "close": current[3] / 4,
                "next_open": following[0] / 4, "next_high": following[1] / 4,
                "next_low": following[2] / 4, "next_close": following[3] / 4,
            }
            for name, value in raw_values.items():
                day_mismatches[name] += int(not np.isclose(float(getattr(row, name)), value, atol=1e-9))
            records.append({"row_id": int(row.Index), **{f"raw_{k}": v for k, v in raw_values.items()}})
        audit_rows.append({
            "year": year, "local_date": date, "bars": len(day),
            "raw_rows": integrity["rows"], "raw_originally_sorted": integrity["originally_sorted"],
            "raw_reordered_rows": integrity["reordered_rows"],
            **{f"mismatch_{k}": v for k, v in day_mismatches.items()},
        })

    corrected = frame.join(pd.DataFrame(records).set_index("row_id"))
    for name in ("open", "high", "low", "close", "next_open", "next_high", "next_low", "next_close"):
        corrected[name] = corrected[f"raw_{name}"]
    corrected["actual_range_points"] = corrected.high - corrected.low
    corrected["range_ratio"] = corrected.actual_range_points / corrected.predicted_range_points
    corrected["above_prediction"] = corrected.range_ratio.gt(1)
    corrected["next_actual_range_points"] = corrected.next_high - corrected.next_low
    # The next bar uses the next bucket's frozen prediction. The audited source
    # label is retained only after next open/high/low have matched raw ticks.
    return corrected, pd.DataFrame(audit_rows)


def features_from_corrected(data: pd.DataFrame, year: int) -> pd.DataFrame:
    data = data.copy().sort_values(["local_date", "bucket"]).reset_index(drop=True)
    data["year"] = year
    data["time"] = pd.to_datetime(data.bucket_start_dt, utc=True)
    candle_range = (data.high - data.low).replace(0, np.nan)
    body = data.close - data.open
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
    data["direction"] = np.sign(body).replace(0, 1).astype(int)
    data["above_prediction"] = data.above_prediction.astype(int)
    keys = data.groupby("local_date", group_keys=False)
    for lag in (1, 2, 3):
        data[f"range_ratio_lag{lag}"] = keys.range_ratio.shift(lag)
        data[f"direction_lag{lag}"] = keys.direction.shift(lag)
        data[f"above_lag{lag}"] = keys.above_prediction.shift(lag)
    data["above_streak"] = keys.above_prediction.transform(
        lambda values: values.astype(int).groupby((~values.astype(bool)).cumsum()).cumsum()
    )
    data["direction_streak"] = keys.direction.transform(consecutive_direction)
    data["rolling4_above_count"] = keys.above_prediction.transform(lambda x: x.rolling(4, min_periods=1).sum())
    data["rolling4_direction_sum"] = keys.direction.transform(lambda x: x.rolling(4, min_periods=1).sum())
    data["rolling4_mean_ratio"] = keys.range_ratio.transform(lambda x: x.rolling(4, min_periods=1).mean())
    count = keys.cumcount().add(1)
    data["session_above_share"] = keys.above_prediction.cumsum() / count
    session_open = keys.open.transform("first")
    data["session_net_to_prediction"] = (data.close - session_open) / predicted
    data["target_next_expansive"] = data.next_above_prediction.astype(int)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data


def model() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=.25, max_iter=2000, class_weight="balanced")),
    ])


def classification_row(year: int, y: np.ndarray, p: np.ndarray) -> dict:
    pred = p >= .5
    high = (p >= .70) | (p <= .30)
    high_correct = np.where(p[high] >= .70, y[high] == 1, y[high] == 0)
    return {
        "year": year, "bars": len(y), "base_expansive_pct": 100 * y.mean(),
        "auc": roc_auc_score(y, p), "accuracy_pct": 100 * accuracy_score(y, pred),
        "balanced_accuracy_pct": 100 * balanced_accuracy_score(y, pred),
        "brier": brier_score_loss(y, p),
        "high_conf_signals": int(high.sum()), "high_conf_coverage_pct": 100 * high.mean(),
        "high_conf_accuracy_pct": 100 * high_correct.mean(),
        "expansive_signals": int((p >= .70).sum()),
        "expansive_accuracy_pct": 100 * y[p >= .70].mean(),
        "contractive_signals": int((p <= .30).sum()),
        "contractive_accuracy_pct": 100 * (1 - y[p <= .30]).mean(),
    }


def range_row(year: int, frame: pd.DataFrame) -> dict:
    error = frame.actual_range_points - frame.predicted_range_points
    relative = error.abs() / frame.predicted_range_points
    return {
        "year": year, "bars": len(frame), "days": frame.local_date.nunique(),
        "median_predicted_points": frame.predicted_range_points.median(),
        "median_actual_points": frame.actual_range_points.median(),
        "median_signed_error_points": error.median(),
        "median_absolute_error_points": error.abs().median(),
        "within_25pct_pct": 100 * relative.le(.25).mean(),
        "within_50pct_pct": 100 * relative.le(.50).mean(),
        "pearson": frame.actual_range_points.corr(frame.predicted_range_points),
        "spearman": frame.actual_range_points.corr(frame.predicted_range_points, method="spearman"),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    corrected, audits = {}, []
    for year in (2024, 2025, 2026):
        frame, audit = audit_and_correct(year)
        corrected[year] = features_from_corrected(frame, year)
        audits.append(audit)
        frame.to_parquet(OUT / f"audited_events_{year}.parquet", index=False)
    audit = pd.concat(audits, ignore_index=True)
    audit.to_csv(OUT / "daily_ohlc_audit.csv", index=False)

    critical = ["open", "high", "low", "next_open", "next_high", "next_low"]
    mismatch_totals = {name: int(audit[f"mismatch_{name}"].sum()) for name in critical}
    if any(mismatch_totals.values()):
        raise RuntimeError(f"Range-defining OHLC mismatches found: {mismatch_totals}")

    fitted = model().fit(corrected[2024][FEATURES], corrected[2024].target_next_expansive)
    class_rows, range_rows, predictions = [], [], []
    for year in (2025, 2026):
        frame = corrected[year]
        y = frame.target_next_expansive.to_numpy()
        p = fitted.predict_proba(frame[FEATURES])[:, 1]
        class_rows.append(classification_row(year, y, p))
        range_rows.append(range_row(year, frame))
        predictions.append(pd.DataFrame({
            "year": year, "local_date": frame.local_date, "bucket": frame.bucket,
            "actual_expansive": y, "probability_expansive": p,
        }))
    classification = pd.DataFrame(class_rows)
    range_accuracy = pd.DataFrame(range_rows)
    classification.to_csv(OUT / "verified_logistic_accuracy.csv", index=False)
    range_accuracy.to_csv(OUT / "verified_numeric_range_accuracy.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "verified_predictions.csv", index=False)
    pd.DataFrame([{
        "audited_days": int(len(audit)), "audited_bars": int(audit.bars.sum()),
        "raw_files_originally_unsorted": int((~audit.raw_originally_sorted).sum()),
        **{f"total_mismatch_{name}": int(audit[f"mismatch_{name}"].sum()) for name in (
            "open", "high", "low", "close", "next_open", "next_high", "next_low", "next_close"
        )},
    }]).to_csv(OUT / "audit_summary.csv", index=False)
    print("AUDIT")
    print(pd.read_csv(OUT / "audit_summary.csv").to_string(index=False))
    print("\nLOGISTIC")
    print(classification.to_string(index=False))
    print("\nNUMERIC RANGE")
    print(range_accuracy.to_string(index=False))


if __name__ == "__main__":
    main()
