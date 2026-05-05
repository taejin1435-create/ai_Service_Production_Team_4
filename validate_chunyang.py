import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from integration import (
    AerationPredictor,
    LSTM_WINDOW_THRESHOLD,
    MIN_T_SIGNAL,
    MAX_T_SIGNAL,
    parse_datetime,
    validate_on_december,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ENCODING = "cp949"


def validate_on_chunyang(predictor: AerationPredictor) -> dict:
    xgb_path  = DATA_DIR / "춘양_12월_xgb.csv"
    lstm_path = DATA_DIR / "춘양_12월_lstm.csv"

    for p in [xgb_path, lstm_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p.name} 없음 — 먼저 실행: py preprocessing/preprocess_chunyang.py"
            )

    df_xgb = pd.read_csv(xgb_path, encoding=ENCODING)
    df_xgb["수집시간"] = parse_datetime(df_xgb["수집시간"])

    df_lstm = pd.read_csv(lstm_path, encoding=ENCODING)
    df_lstm["수집시간"] = parse_datetime(df_lstm["수집시간"])
    df_lstm = df_lstm[df_lstm["current_aeration"] == 1].copy().reset_index(drop=True)
    df_lstm["cycle_id"] = (df_lstm["elapsed_time"] == 0).cumsum()

    xgb_starts = (
        df_xgb[df_xgb["elapsed_time"] == 0]
        .set_index(["수집시간", "계측기명"])
    )

    t_signals, t_trues, elapsed_vals = [], [], []
    n_cycles = 0
    actual_total_min = predicted_total_min = 0.0

    for _, cycle in df_lstm.groupby("cycle_id"):
        start = cycle.iloc[0]
        key   = (start["수집시간"], start["계측기명"])
        if key not in xgb_starts.index:
            continue

        xgb_row = xgb_starts.loc[key]
        if isinstance(xgb_row, pd.DataFrame):
            xgb_row = xgb_row.iloc[0]

        actual_runtime = float(xgb_row.get("cycle_runtime", 0))
        predictor.on_cycle_start(xgb_row)
        actual_total_min    += actual_runtime
        predicted_total_min += float(predictor._cycle_runtime_xgb)
        n_cycles += 1

        for _, row in cycle.iterrows():
            raw_elapsed = int(row["elapsed_time"])
            t_signal    = predictor.predict(row, raw_elapsed)
            t_signals.append(t_signal)
            t_trues.append(float(row["T_remaining"]))
            elapsed_vals.append(raw_elapsed)

    t_signals    = np.array(t_signals,    dtype=np.float32)
    t_trues      = np.array(t_trues,      dtype=np.float32)
    elapsed_vals = np.array(elapsed_vals, dtype=np.float32)

    mae  = float(mean_absolute_error(t_trues, t_signals))
    rmse = float(np.sqrt(mean_squared_error(t_trues, t_signals)))

    stages = []
    for label, lo, hi in [("0~30분", 0, 30), ("31~90분", 31, 90), ("91분+", 91, 9999)]:
        mask = (elapsed_vals >= lo) & (elapsed_vals <= hi)
        if mask.sum() == 0:
            continue
        stages.append({
            "label": label,
            "mae":   round(float(mean_absolute_error(t_trues[mask], t_signals[mask])), 2),
            "n":     int(mask.sum()),
        })

    mask_91       = elapsed_vals >= 91
    stage_91_mae  = round(float(mean_absolute_error(t_trues[mask_91], t_signals[mask_91])), 2) if mask_91.sum() > 0 else None
    stage_91_rmse = round(float(np.sqrt(mean_squared_error(t_trues[mask_91], t_signals[mask_91]))), 2) if mask_91.sum() > 0 else None

    return {
        "mae":          round(mae,  2),
        "rmse":         round(rmse, 2),
        "stage_91_mae": stage_91_mae,
        "stage_91_rmse":stage_91_rmse,
        "stages":       stages,
        "n_cycles":     n_cycles,
        "n_steps":      len(t_signals),
    }


def print_comparison(bongwha: dict, chunyang: dict) -> None:
    target_mae, target_rmse = 30, 45

    def status(val, target):
        return "PASS ✅" if val is not None and val < target else "FAIL ❌"

    print("\n" + "=" * 60)
    print("  교차 처리장 검증 — 봉화(학습) vs 춘양(미학습)")
    print("=" * 60)
    print(f"{'':20} {'봉화 12월':>15} {'춘양 12월':>15}")
    print("-" * 60)
    print(f"{'전체 MAE':20} {bongwha['mae']:>14.2f}분 {chunyang['mae']:>14.2f}분")
    print(f"{'전체 RMSE':20} {bongwha['rmse']:>14.2f}분 {chunyang['rmse']:>14.2f}분")
    print("-" * 60)
    print(f"{'[종료 판단] 91분+ MAE':20} {bongwha['stage_91_mae']:>14.2f}분 {str(chunyang['stage_91_mae'] or '-'):>14}분")
    print(f"{'[종료 판단] 91분+ RMSE':20} {bongwha['stage_91_rmse']:>14.2f}분 {str(chunyang['stage_91_rmse'] or '-'):>14}분")
    print(f"{'목표 (MAE < 30분)':20} {status(bongwha['stage_91_mae'], target_mae):>15} {status(chunyang['stage_91_mae'], target_mae):>15}")
    print(f"{'목표 (RMSE < 45분)':20} {status(bongwha['stage_91_rmse'], target_rmse):>15} {status(chunyang['stage_91_rmse'], target_rmse):>15}")
    print("-" * 60)
    print(f"{'사이클 수':20} {bongwha['n_cycles']:>15} {chunyang['n_cycles']:>15}")
    print(f"{'총 스텝 수':20} {bongwha['n_steps']:>15} {chunyang['n_steps']:>15}")
    print("=" * 60)

    print("\n  구간별 MAE 상세")
    print(f"{'구간':10} {'봉화':>12} {'춘양':>12}")
    print("-" * 36)

    bh_stage = {s["label"]: s for s in bongwha["stages"]}
    cy_stage = {s["label"]: s for s in chunyang["stages"]}

    for label in ["0~30분", "31~90분", "91분+"]:
        bh = bh_stage.get(label)
        cy = cy_stage.get(label)
        bh_str = f"{bh['mae']:.2f}분 (n={bh['n']})" if bh else "-"
        cy_str = f"{cy['mae']:.2f}분 (n={cy['n']})" if cy else "-"
        print(f"{label:10} {bh_str:>12} {cy_str:>12}")

    print()


def main():
    print("모델 로드 중...")
    predictor_bh = AerationPredictor()
    predictor_cy = AerationPredictor()

    print("봉화 12월 검증 중...")
    bongwha = validate_on_december(predictor_bh)
    bongwha["n_steps"]  = sum(s["n"] for s in bongwha.get("stages", []))
    bongwha["n_cycles"] = bongwha.get("energy", {}).get("test_cycles", 0)

    print("\n춘양 12월 검증 중...")
    chunyang = validate_on_chunyang(predictor_cy)

    print_comparison(bongwha, chunyang)


if __name__ == "__main__":
    main()
