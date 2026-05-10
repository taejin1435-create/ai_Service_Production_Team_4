import json
import pickle
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import XGB_FEATURES, LSTM_FEATURES
from utils import parse_datetime, underprediction_rate, update_metadata, generate_report

warnings.filterwarnings("ignore")

BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
MODELS_DIR      = BASE_DIR / "models"
XGB_MODEL_PATH  = MODELS_DIR / "xgboost_model.json"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.pt"
SCALER_PATH     = MODELS_DIR / "lstm_scaler.pkl"
ENCODING        = "cp949"

WINDOW_SIZE           = 7
LSTM_WINDOW_THRESHOLD = (WINDOW_SIZE - 1) * 10  # 60분: LSTM 활성화 시작
MIN_T_SIGNAL          = 10
STOP_THRESHOLD        = 20
EARLY_STOP_MARGIN     = 20
MAX_T_SIGNAL          = 250
NH4_SAFETY_THRESHOLD  = 5.0
HIDDEN_SIZE           = 128
NUM_LAYERS            = 2
DROPOUT               = 0.3

AERATOR_KW_PER_MIN = 958_862 * 0.55 / 365 / 1440
KRW_PER_KWH        = 110
DECEMBER_DAYS      = 31


class CycleRemainingLSTM(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class AerationPredictor:
    def __init__(self):
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(str(XGB_MODEL_PATH))

        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

        self.lstm_model = CycleRemainingLSTM(input_size=len(LSTM_FEATURES))
        self.lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, weights_only=True))
        self.lstm_model.eval()

        self._buffer: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
        self._cycle_runtime_xgb: int = 0
        self._log: list[dict] = []
        self._last_elapsed: int = -1

    def on_cycle_start(self, xgb_row: pd.Series) -> int:
        self._buffer.clear()
        self._log.clear()
        self._last_elapsed = -1
        pred = self.xgb_model.predict(pd.DataFrame([xgb_row[XGB_FEATURES]]))[0]
        self._cycle_runtime_xgb = int(max(MIN_T_SIGNAL, pred))
        return self._cycle_runtime_xgb

    def predict(self, lstm_row: pd.Series, raw_elapsed: int) -> tuple[int, bool]:
        if raw_elapsed < self._last_elapsed:
            self._buffer.clear()
        self._last_elapsed = raw_elapsed

        scaled = self.scaler.transform(
            pd.DataFrame([lstm_row[LSTM_FEATURES]])
        )[0].astype(np.float32)
        scaled = np.clip(scaled, -5.0, 5.0)
        self._buffer.append(scaled)

        xgb_remain = max(0, self._cycle_runtime_xgb - raw_elapsed)

        if raw_elapsed < LSTM_WINDOW_THRESHOLD or len(self._buffer) < WINDOW_SIZE:
            t_signal = float(xgb_remain)
            mode = "xgb_only"
        else:
            window = np.array(list(self._buffer), dtype=np.float32)[np.newaxis]
            with torch.no_grad():
                lstm_remain = float(
                    max(0.0, self.lstm_model(torch.tensor(window)).item())
                )
            t_signal = max(xgb_remain, lstm_remain)
            mode = "integrated"

        raw_t_signal = t_signal
        t_signal = int(np.clip(t_signal, MIN_T_SIGNAL, MAX_T_SIGNAL))
        was_clamped = not (MIN_T_SIGNAL <= raw_t_signal <= MAX_T_SIGNAL)
        self._log.append({
            "elapsed_time":       raw_elapsed,
            "t_signal":           t_signal,
            "predicted_end_time": raw_elapsed + t_signal,
            "xgb_prediction":     self._cycle_runtime_xgb,
            "mode":               mode,
            "was_clamped":        was_clamped,
        })
        return t_signal, was_clamped

    def should_continue(self, t_signal: int) -> bool:
        return t_signal > STOP_THRESHOLD

    def get_history(self) -> dict:
        return {
            "cycle_runtime_xgb": self._cycle_runtime_xgb,
            "log": list(self._log),
        }

def validate_on_december(predictor: AerationPredictor) -> dict:
    df_xgb = pd.read_csv(DATA_DIR / "봉화_12월_xgb.csv", encoding=ENCODING)
    df_xgb["수집시간"] = parse_datetime(df_xgb["수집시간"])

    df_lstm = pd.read_csv(DATA_DIR / "봉화_12월_lstm.csv", encoding=ENCODING)
    df_lstm["수집시간"] = parse_datetime(df_lstm["수집시간"])
    df_lstm = df_lstm[df_lstm["current_aeration"] == 1].copy().reset_index(drop=True)
    df_lstm["cycle_id"] = (df_lstm["elapsed_time"] == 0).cumsum()

    xgb_starts = (
        df_xgb[df_xgb["elapsed_time"] == 0]
        .set_index(["수집시간", "계측기명"])
    )

    t_signals, t_trues, elapsed_vals = [], [], []
    stop_nh4_vals     = []
    stop_elapsed_vals = []
    actual_runtime_vals = []
    n_xgb_only = n_integrated = n_clamped_xgb = n_clamped_integrated = 0
    actual_total_min = 0.0
    n_cycles = 0

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
        actual_total_min += actual_runtime
        n_cycles += 1

        stop_elapsed = None
        stop_nh4     = None
        for _, row in cycle.iterrows():
            raw_elapsed = int(row["elapsed_time"])
            t_signal, was_clamped = predictor.predict(row, raw_elapsed)
            t_true                = float(row["T_remaining"])

            t_signals.append(t_signal)
            t_trues.append(t_true)
            elapsed_vals.append(raw_elapsed)

            if stop_elapsed is None and t_signal <= STOP_THRESHOLD:
                stop_elapsed = raw_elapsed
                stop_nh4     = float(row["nh4"])

            if raw_elapsed < LSTM_WINDOW_THRESHOLD:
                n_xgb_only += 1
                if was_clamped:
                    n_clamped_xgb += 1
            else:
                n_integrated += 1
                if was_clamped:
                    n_clamped_integrated += 1

        actual_runtime_vals.append(actual_runtime)
        stop_elapsed_vals.append(stop_elapsed)
        if stop_nh4 is not None:
            stop_nh4_vals.append(stop_nh4)

    t_signals    = np.array(t_signals,    dtype=np.float32)
    t_trues      = np.array(t_trues,      dtype=np.float32)
    elapsed_vals = np.array(elapsed_vals, dtype=np.float32)

    mae   = mean_absolute_error(t_trues, t_signals)
    rmse  = float(np.sqrt(mean_squared_error(t_trues, t_signals)))
    under = underprediction_rate(t_signals, t_trues)

    stop_nh4_arr  = np.array(stop_nh4_vals, dtype=np.float32)
    nh4_risk_rate = float((stop_nh4_arr > NH4_SAFETY_THRESHOLD).mean()) if len(stop_nh4_arr) > 0 else 0.0

    actual_arr        = np.array(actual_runtime_vals, dtype=np.float32)
    stopped_mask      = np.array([s is not None for s in stop_elapsed_vals])
    stop_elapsed_fill = np.array(
        [s if s is not None else np.nan for s in stop_elapsed_vals], dtype=np.float32
    )
    time_diff    = actual_arr - stop_elapsed_fill
    stopped_diff = time_diff[stopped_mask]

    no_stop_rate    = float((~stopped_mask).mean())
    avg_time_saved  = float(np.nanmean(stopped_diff))              if stopped_mask.sum() > 0 else 0.0
    unsafe_early_stop_rate = float((stopped_diff > EARLY_STOP_MARGIN).mean()) if stopped_mask.sum() > 0 else 0.0
    avg_overrun     = float(np.maximum(0.0, -stopped_diff).mean()) if stopped_mask.sum() > 0 else 0.0

    print("\n=== 통합 모델 평가 (12월 TEST) ===")
    print(f"MAE:  {mae:.2f}분 | RMSE: {rmse:.2f}분 | 심각한 과소예측: {under*100:.1f}% | n={len(t_signals)}")
    print(f"수질 불량 위험 사이클: {nh4_risk_rate*100:.1f}%  (STOP 시점 NH4 > {NH4_SAFETY_THRESHOLD} mg/L, n={len(stop_nh4_arr)})")

    print("\n=== 운영 STOP 시뮬레이션 ===")
    print(f"  평균 절감 시간  : {avg_time_saved:+.1f}분/사이클  (+ = 절감, - = 초과 가동)")
    print(f"  위험 조기종료율 : {unsafe_early_stop_rate*100:.1f}%  (실제 종료 20분+ 전 STOP)")
    print(f"  평균 초과 가동  : {avg_overrun:.1f}분  (늦게 끈 사이클 기준)")
    print(f"  STOP 미발생율   : {no_stop_rate*100:.1f}%  (n={int((~stopped_mask).sum())}사이클)")

    print("\n=== 구간별 MAE ===")
    for label, lo, hi in [
        ("초반  0~30분",  0,  30),
        ("중반 31~90분", 31,  90),
        ("후반  91분+",  91, 9999),
    ]:
        mask = (elapsed_vals >= lo) & (elapsed_vals <= hi)
        if mask.sum() == 0:
            continue
        print(f"  {label}: MAE {mean_absolute_error(t_trues[mask], t_signals[mask]):.2f}분  (n={mask.sum()})")

    integ_mask = elapsed_vals >= LSTM_WINDOW_THRESHOLD
    integ_mae  = mean_absolute_error(t_trues[integ_mask], t_signals[integ_mask])
    integ_rmse = float(np.sqrt(mean_squared_error(t_trues[integ_mask], t_signals[integ_mask])))

    meta      = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8")) if (MODELS_DIR / "metadata.json").exists() else {}
    lstm_mae  = meta.get("metrics", {}).get("lstm_mae",  "—")
    lstm_rmse = meta.get("metrics", {}).get("lstm_rmse", "—")
    print(f"\n=== 공정 비교: 통합 구간만 (elapsed ≥ {LSTM_WINDOW_THRESHOLD}분) ===")
    print(f"  MAE: {integ_mae:.2f}분 | RMSE: {integ_rmse:.2f}분 | n={integ_mask.sum()}")
    print(f"  (참고) LSTM 단독 MAE: {lstm_mae}분 / RMSE: {lstm_rmse}분")

    n_clamped = n_clamped_xgb + n_clamped_integrated
    print("\n=== 모델 기여 분석 ===")
    print(f"  XGBoost 단독 (elapsed < {LSTM_WINDOW_THRESHOLD}분): {n_xgb_only:,}개 timestep")
    print(f"  통합 신호    (elapsed ≥ {LSTM_WINDOW_THRESHOLD}분): {n_integrated:,}개 timestep")
    if n_clamped:
        print(f"  Safety Clamping: 총 {n_clamped}회 "
              f"(XGB구간 {n_clamped_xgb}회 / 통합구간 {n_clamped_integrated}회)")

    stages = []
    for label, lo, hi in [
        ("0~30분",   0,  30),
        ("31~90분", 31,  90),
        ("91분+",   91, 9999),
    ]:
        mask = (elapsed_vals >= lo) & (elapsed_vals <= hi)
        if mask.sum() == 0:
            continue
        stages.append({
            "label": label,
            "mae":   round(float(mean_absolute_error(t_trues[mask], t_signals[mask])), 2),
            "n":     int(mask.sum()),
        })

    mask_91        = elapsed_vals >= 91
    stage_91_mae   = round(float(mean_absolute_error(t_trues[mask_91], t_signals[mask_91])), 2) if mask_91.sum() > 0 else None
    stage_91_rmse  = round(float(np.sqrt(mean_squared_error(t_trues[mask_91], t_signals[mask_91]))), 2) if mask_91.sum() > 0 else None

    annual_scale       = 365 / DECEMBER_DAYS
    sim_stop_arr       = np.where(stopped_mask, stop_elapsed_fill, actual_arr)
    saved_min_sim      = float(np.nansum(actual_arr - sim_stop_arr))
    saved_kwh_sim      = saved_min_sim * annual_scale * AERATOR_KW_PER_MIN
    saved_krw_sim      = saved_kwh_sim * KRW_PER_KWH

    energy = {
        "source":              "봉화읍 하수처리장 (경상북도 2023)",
        "annual_total_kwh":    958_862,
        "aerator_ratio":       0.55,
        "kwh_per_min":         round(AERATOR_KW_PER_MIN, 4),
        "krw_per_kwh":         KRW_PER_KWH,
        "test_cycles":         n_cycles,
        "actual_avg_min":      round(actual_total_min / n_cycles, 1) if n_cycles else 0,
        "sim_stop_avg_min":    round(float(np.nanmean(sim_stop_arr)), 1),
        "avg_time_saved":      round(avg_time_saved, 1),
        "saved_kwh_annual":    round(saved_kwh_sim),
        "saved_krw_annual":    round(saved_krw_sim),
    }

    print(f"\n=== 전력 절감 추정 (시뮬레이션 기반, 12월 → 연간 환산) ===")
    print(f"  실제 평균 사이클 시간  : {energy['actual_avg_min']}분")
    print(f"  시뮬레이션 STOP 평균   : {energy['sim_stop_avg_min']}분")
    print(f"  사이클당 평균 절감     : {avg_time_saved:+.1f}분")
    print(f"  연간 절감 전력         : {energy['saved_kwh_annual']:,} kWh")
    print(f"  연간 절감 전기료       : {energy['saved_krw_annual']:,} 원")

    stop_kpi = {
        "avg_time_saved":  round(avg_time_saved,  1),
        "unsafe_early_stop_rate": round(unsafe_early_stop_rate, 4),
        "avg_overrun":     round(avg_overrun,      1),
        "no_stop_rate":    round(no_stop_rate,     4),
        "nh4_risk_rate":   round(nh4_risk_rate,    4),
    }

    update_metadata({
        "metrics": {
            "integration_mae":      round(float(mae),         2),
            "integration_rmse":     round(float(rmse),        2),
            "underprediction_rate": round(float(under),       4),
            "nh4_risk_rate":        round(float(nh4_risk_rate), 4),
            "avg_time_saved":       round(avg_time_saved,      1),
            "unsafe_early_stop_rate":      round(unsafe_early_stop_rate,     4),
            "avg_overrun":          round(avg_overrun,          1),
            "no_stop_rate":         round(no_stop_rate,         4),
        },
    }, MODELS_DIR)
    generate_report(MODELS_DIR)

    return {
        "mae":                  mae,
        "rmse":                 rmse,
        "underprediction_rate": under,
        "nh4_risk_rate":        nh4_risk_rate,
        "stage_91_mae":         stage_91_mae,
        "stage_91_rmse":        stage_91_rmse,
        "stages":               stages,
        "energy":               energy,
        "stop_kpi":             stop_kpi,
        "n_cycles":             n_cycles,
        "n_steps":              len(t_signals),
    }


def main() -> None:
    print("모델 로드 중...")
    predictor = AerationPredictor()
    print("  XGBoost ✓  |  LSTM ✓  |  Scaler ✓")

    metrics = validate_on_december(predictor)

    mae_91  = metrics["stage_91_mae"]
    rmse_91 = metrics["stage_91_rmse"]
    mae_status  = "PASS ✅" if mae_91  is not None and mae_91  < 30 else "FAIL ❌"
    rmse_status = "PASS ✅" if rmse_91 is not None and rmse_91 < 45 else "FAIL ❌"
    print(f"\n목표 대비 [종료 판단 구간 91분+]: "
          f"MAE {mae_status} ({mae_91}분 / 목표 <30분) | "
          f"RMSE {rmse_status} ({rmse_91}분 / 목표 <45분)")


if __name__ == "__main__":
    main()
