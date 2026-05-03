import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "xgboost_model.json"
ENCODING = "cp949"
MONTHS = [8, 9, 10, 12]
DATE_FORMAT = "%m월%d일%y %H:%M"
TRAIN_END = pd.Timestamp("2020-12-01")

FEATURES = [
    "nh4",
    "no3",
    "ph",
    "temp",
    "상전류(R)",
    "nh4_no3_ratio",
    "do_saturation",
    "nh4_diff",
    "no3_diff",
    "ph_diff",
    "nh4_rolling_mean",
    "nh4_decay_rate",
    "hour_sin",
    "hour_cos",
    "weekday",
    "reactor",
]

MONOTONE_CONSTRAINTS = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)

TARGET = "cycle_runtime"
FORBIDDEN_FEATURES = {"elapsed_time", "T_remaining", "current_aeration"}


def assert_pipeline_integrity(features: list[str], constraints: tuple) -> None:
    leaks = set(features) & FORBIDDEN_FEATURES
    if leaks:
        raise ValueError(f"Leakage features detected: {leaks}")
    if len(features) != len(constraints):
        raise ValueError(
            f"FEATURES({len(features)}) ↔ MONOTONE_CONSTRAINTS({len(constraints)}) 길이 불일치"
        )
    if len(set(features)) != len(features):
        raise ValueError("FEATURES 중복 존재")


def parse_datetime(series: pd.Series) -> pd.Series:
    result = pd.to_datetime(series, format="%m월%d월%y %H:%M", errors="coerce")
    mask = result.isna()
    if mask.any():
        result[mask] = pd.to_datetime(series[mask], errors="coerce")
    return result


def load_xgb_data(months: list[int]) -> pd.DataFrame:
    frames = []
    for m in months:
        path = DATA_DIR / f"봉화_{m}월_xgb.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path, encoding=ENCODING))
    df = pd.concat(frames, ignore_index=True)
    df["수집시간"] = parse_datetime(df["수집시간"])
    if df["수집시간"].isna().any():
        n_bad = df["수집시간"].isna().sum()
        raise ValueError(f"수집시간 파싱 실패 행 {n_bad}건")
    return df.sort_values("수집시간").reset_index(drop=True)


def filter_cycle_starts(df: pd.DataFrame) -> pd.DataFrame:
    if "elapsed_time" not in df.columns:
        raise KeyError("elapsed_time 컬럼 누락 — 전처리 단계 확인 필요")
    df = df[df["elapsed_time"] == 0].copy()
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    return df


def split_by_month(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["수집시간"] < TRAIN_END].reset_index(drop=True)
    test = df[df["수집시간"] >= TRAIN_END].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("month-based split 결과 한쪽이 비어있음")
    return train, test


def build_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        gamma=1.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        monotone_constraints=MONOTONE_CONSTRAINTS,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


def predict_clipped(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    return np.maximum(0.0, model.predict(X))


def evaluate(model: xgb.XGBRegressor, X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    pred = predict_clipped(model, X)
    mae = mean_absolute_error(y, pred)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    print(f"[{label}] MAE: {mae:6.2f}분 | RMSE: {rmse:6.2f}분 | n={len(y)}")
    return {"mae": mae, "rmse": rmse, "n": len(y)}


def report_importance(model: xgb.XGBRegressor) -> pd.DataFrame:
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    df = pd.DataFrame(
        {"gain": [gain.get(f, 0.0) for f in FEATURES],
         "weight": [weight.get(f, 0.0) for f in FEATURES]},
        index=FEATURES,
    ).sort_values("gain", ascending=False)
    print("\n=== Feature Importance (gain | weight) ===")
    print(df.to_string())
    top1, top2 = df["gain"].iloc[0], df["gain"].iloc[1] if len(df) > 1 else 0.0
    if top2 > 0 and top1 / top2 >= 3.0:
        print(f"⚠ 지배적 피처 의심: {df.index[0]} (gain {top1:.2f} ≥ 2위 × 3)")
    return df


def report_shap(model: xgb.XGBRegressor, X_sample: pd.DataFrame) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_sample)
    abs_mean = np.abs(values).mean(axis=0)
    df = pd.DataFrame({"shap_abs_mean": abs_mean}, index=FEATURES)
    df = df.sort_values("shap_abs_mean", ascending=False)
    print("\n=== SHAP Mean |Value| ===")
    print(df.to_string())
    quality_features = {"nh4", "no3", "nh4_diff", "no3_diff", "nh4_no3_ratio", "nh4_decay_rate", "nh4_rolling_mean"}
    top5 = set(df.head(5).index)
    if not (quality_features & top5):
        print("⚠ 수질 피처가 SHAP 상위 5위 내 부재 — 모델 재검토 권장")
    return df


def main() -> None:
    assert_pipeline_integrity(FEATURES, MONOTONE_CONSTRAINTS)

    df = load_xgb_data(MONTHS)
    df = filter_cycle_starts(df)

    train, test = split_by_month(df)
    print("=== 샘플 수 (cycle starts) ===")
    for m in MONTHS:
        label = f"{'train' if m != 12 else 'test ':5}"
        n = df[df["수집시간"].dt.month == m]
        print(f"  {m:2d}월 [{label}]: {len(n):>4d}개")
    print(f"  합계       train: {len(train):>4d}개 | test: {len(test):>4d}개 | 총: {len(df):>4d}개")
    print(f"train 기간: {train['수집시간'].min()} ~ {train['수집시간'].max()}")
    print(f"test  기간: {test['수집시간'].min()} ~ {test['수집시간'].max()}")
    print(f"\ncycle_runtime 분포 (train):\n{train[TARGET].describe()}\n")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    model = build_model()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50,
    )

    print("\n=== 평가 ===")
    evaluate(model, X_train, y_train, "TRAIN")
    test_metrics = evaluate(model, X_test, y_test, "TEST ")

    report_importance(model)
    sample_size = min(1000, len(X_test))
    sample = X_test.sample(sample_size, random_state=42) if sample_size > 0 else X_test
    report_shap(model, sample)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"\n모델 저장: {MODEL_PATH}")
    print(f"목표 대비: MAE {'PASS' if test_metrics['mae'] < 60 else 'FAIL'} (<60) | "
          f"RMSE {'PASS' if test_metrics['rmse'] < 75 else 'FAIL'} (<75)")


if __name__ == "__main__":
    main()
