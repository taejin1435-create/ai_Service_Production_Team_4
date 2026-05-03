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

warnings.filterwarnings("ignore")

BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
XGB_MODEL_PATH  = BASE_DIR / "xgboost_model.json"
LSTM_MODEL_PATH = BASE_DIR / "lstm_model.pt"
SCALER_PATH     = BASE_DIR / "lstm_scaler.pkl"
ENCODING        = "cp949"

WINDOW_SIZE           = 9
LSTM_WINDOW_THRESHOLD = (WINDOW_SIZE - 1) * 10   # 80분: 이 미만은 XGBoost 단독
MIN_T_SIGNAL          = 10
MAX_T_SIGNAL          = 250
HIDDEN_SIZE           = 128
NUM_LAYERS            = 2
DROPOUT               = 0.3

XGB_FEATURES = [
    "nh4", "no3", "ph", "temp", "상전류(R)",
    "nh4_no3_ratio", "do_saturation",
    "nh4_diff", "no3_diff", "ph_diff",
    "nh4_rolling_mean", "nh4_decay_rate",
    "hour_sin", "hour_cos", "weekday", "reactor",
]

LSTM_FEATURES = [
    "nh4", "no3", "ph", "temp", "상전류(R)",
    "nh4_no3_ratio", "do_saturation",
    "nh4_diff", "no3_diff", "ph_diff",
    "nh4_rolling_mean", "nh4_decay_rate",
    "hour_sin", "hour_cos", "weekday",
    "elapsed_time",
]


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


def parse_datetime(series: pd.Series) -> pd.Series:
    result = pd.to_datetime(series, format="%m월%d월%y %H:%M", errors="coerce")
    mask = result.isna()
    if mask.any():
        result[mask] = pd.to_datetime(series[mask], errors="coerce")
    return result


class AerationPredictor:
    def __init__(self):
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(str(XGB_MODEL_PATH))

        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

        self.lstm_model = CycleRemainingLSTM(input_size=len(LSTM_FEATURES))
        self.lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, weights_only=True))
        self.lstm_model.eval()

        self._buffer = deque(maxlen=WINDOW_SIZE)
        self._cycle_runtime_xgb: int = 0
        self._log: list[dict] = []

    def on_cycle_start(self, xgb_row: pd.Series) -> int:
        self._buffer.clear()
        self._log.clear()
        pred = self.xgb_model.predict(pd.DataFrame([xgb_row[XGB_FEATURES]]))[0]
        self._cycle_runtime_xgb = int(max(0, pred))
        return self._cycle_runtime_xgb

    def predict(self, lstm_row: pd.Series, raw_elapsed: int) -> int:
        scaled = self.scaler.transform(
            pd.DataFrame([lstm_row[LSTM_FEATURES]])
        )[0].astype(np.float32)
        self._buffer.append(scaled)

        xgb_remain = self._cycle_runtime_xgb - raw_elapsed

        if raw_elapsed < LSTM_WINDOW_THRESHOLD or len(self._buffer) < WINDOW_SIZE:
            t_signal = xgb_remain
            mode = "xgb_only"
        else:
            window = np.array(list(self._buffer), dtype=np.float32)[np.newaxis]
            with torch.no_grad():
                lstm_remain = float(
                    max(0.0, self.lstm_model(torch.tensor(window)).item())
                )
            t_signal = max(xgb_remain, lstm_remain)
            mode = "integrated"

        t_signal = int(np.clip(t_signal, MIN_T_SIGNAL, MAX_T_SIGNAL))
        self._log.append({
            "elapsed_time":       raw_elapsed,
            "t_signal":           t_signal,
            "predicted_end_time": raw_elapsed + t_signal,
            "xgb_prediction":     self._cycle_runtime_xgb,
            "mode":               mode,
        })
        return t_signal

    def get_history(self) -> dict:
        return {
            "cycle_runtime_xgb": self._cycle_runtime_xgb,
            "log": list(self._log),
        }

    def should_continue(self, raw_elapsed: int) -> bool:
        return (self._cycle_runtime_xgb - raw_elapsed) > 0


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
    n_xgb_only = n_integrated = n_clamped = 0

    for _, cycle in df_lstm.groupby("cycle_id"):
        start = cycle.iloc[0]
        key   = (start["수집시간"], start["계측기명"])

        if key not in xgb_starts.index:
            continue

        xgb_row = xgb_starts.loc[key]
        if isinstance(xgb_row, pd.DataFrame):
            xgb_row = xgb_row.iloc[0]

        predictor.on_cycle_start(xgb_row)

        for _, row in cycle.iterrows():
            raw_elapsed = int(row["elapsed_time"])
            t_signal    = predictor.predict(row, raw_elapsed)
            t_true      = float(row["T_remaining"])

            t_signals.append(t_signal)
            t_trues.append(t_true)
            elapsed_vals.append(raw_elapsed)

            if raw_elapsed < LSTM_WINDOW_THRESHOLD:
                n_xgb_only += 1
            else:
                n_integrated += 1
            if t_signal in (MIN_T_SIGNAL, MAX_T_SIGNAL):
                n_clamped += 1

    t_signals    = np.array(t_signals,    dtype=np.float32)
    t_trues      = np.array(t_trues,      dtype=np.float32)
    elapsed_vals = np.array(elapsed_vals, dtype=np.float32)

    mae  = mean_absolute_error(t_trues, t_signals)
    rmse = float(np.sqrt(mean_squared_error(t_trues, t_signals)))

    print("\n=== 통합 모델 평가 (12월 TEST) ===")
    print(f"MAE:  {mae:.2f}분 | RMSE: {rmse:.2f}분 | n={len(t_signals)}")

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

    print("\n=== 공정 비교: 통합 구간만 (elapsed ≥ 80분) ===")
    print(f"  MAE: {integ_mae:.2f}분 | RMSE: {integ_rmse:.2f}분 | n={integ_mask.sum()}")
    print(f"  (참고) LSTM 단독 MAE: 34.03분 / RMSE: 42.12분 | n=1,650")

    print("\n=== 모델 기여 분석 ===")
    print(f"  XGBoost 단독 (elapsed < {LSTM_WINDOW_THRESHOLD}분): {n_xgb_only:,}개 timestep")
    print(f"  통합 신호    (elapsed ≥ {LSTM_WINDOW_THRESHOLD}분): {n_integrated:,}개 timestep")
    if n_clamped:
        print(f"  Safety Clamping 발동: {n_clamped}회 (주로 사이클 종료 직전 정상 동작)")

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

    return {"mae": mae, "rmse": rmse, "stages": stages}


def main() -> None:
    print("모델 로드 중...")
    predictor = AerationPredictor()
    print("  XGBoost ✓  |  LSTM ✓  |  Scaler ✓")

    metrics = validate_on_december(predictor)

    print(f"\n목표 대비: "
          f"MAE {'PASS' if metrics['mae'] < 30 else 'FAIL'} (<30분) | "
          f"RMSE {'PASS' if metrics['rmse'] < 45 else 'FAIL'} (<45분)")


if __name__ == "__main__":
    main()
