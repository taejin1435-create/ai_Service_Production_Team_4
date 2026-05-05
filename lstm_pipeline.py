import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
MODEL_PATH  = BASE_DIR / "lstm_model.pt"
SCALER_PATH = BASE_DIR / "lstm_scaler.pkl"
ENCODING    = "cp949"

TRAIN_MONTHS  = [7, 8, 9, 10, 11]
TEST_MONTHS   = [12]
WINDOW_SIZE   = 9
BATCH_SIZE    = 64
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
DROPOUT       = 0.3
LEARNING_RATE = 1e-3
MAX_EPOCHS    = 100
PATIENCE      = 15
HUBER_DELTA   = 5.0
SEPT_IMPUTE = {
    "temp":         (23.87, 21.33),   # (8월 말 평균, 10월 초 평균)
    "do_saturation": (8.437, 8.843),
}

TARGET   = "T_remaining"
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
    "elapsed_time",
]
FORBIDDEN = {"cycle_runtime", "current_aeration", "T_remaining"}


def assert_pipeline_integrity(features: list[str]) -> None:
    leaks = set(features) & FORBIDDEN
    if leaks:
        raise ValueError(f"Leakage features detected: {leaks}")
    if len(set(features)) != len(features):
        raise ValueError("FEATURES 중복 존재")



def impute_september(df: pd.DataFrame) -> pd.DataFrame:
    sept_mask = df["수집시간"].dt.month == 9
    if not sept_mask.any():
        return df
    df = df.copy()
    t = (df.loc[sept_mask, "수집시간"].dt.day - 1) / 29
    for col, (v_start, v_end) in SEPT_IMPUTE.items():
        df.loc[sept_mask, col] = v_start + (v_end - v_start) * t
    return df


def parse_datetime(series: pd.Series) -> pd.Series:
    result = pd.to_datetime(series, format="%m월%d월%y %H:%M", errors="coerce")
    mask = result.isna()
    if mask.any():
        result[mask] = pd.to_datetime(series[mask], errors="coerce")
    return result


def load_lstm_data(months: list[int]) -> pd.DataFrame:
    frames = []
    for m in months:
        path = DATA_DIR / f"봉화_{m}월_lstm.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path, encoding=ENCODING))
    df = pd.concat(frames, ignore_index=True)
    df["수집시간"] = parse_datetime(df["수집시간"])
    if df["수집시간"].isna().any():
        raise ValueError(f"수집시간 파싱 실패 {df['수집시간'].isna().sum()}건")
    df = df.sort_values("수집시간").reset_index(drop=True)
    for col in ["temp", "do_saturation"]:
        df[col] = df[col].replace(-999, np.nan).interpolate(method="linear", limit_direction="both")
    df = impute_september(df)
    return df[df["current_aeration"] == 1].copy()


def assign_cycle_id(df: pd.DataFrame, id_offset: int = 0) -> pd.DataFrame:
    df = df.copy()
    df["cycle_id"] = (df["elapsed_time"] == 0).cumsum() + id_offset
    return df


def build_sequences(df: pd.DataFrame, raw_elapsed: pd.Series):
    X_list, y_list, elapsed_list = [], [], []
    for cid, cycle in df.groupby("cycle_id"):
        vals    = cycle[FEATURES].values.astype(np.float32)
        targets = cycle[TARGET].values.astype(np.float32)
        elapsed = raw_elapsed.loc[cycle.index].values
        if len(vals) < WINDOW_SIZE:
            continue
        for i in range(WINDOW_SIZE - 1, len(vals)):
            X_list.append(vals[i - WINDOW_SIZE + 1 : i + 1])
            y_list.append(targets[i])
            elapsed_list.append(elapsed[i])
    return (
        np.array(X_list,      dtype=np.float32),
        np.array(y_list,      dtype=np.float32),
        np.array(elapsed_list, dtype=np.float32),
    )


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.tensor(X), torch.tensor(y)),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


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


def _compute_mae(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds.extend(np.maximum(0.0, model(X_batch).numpy()))
            trues.extend(y_batch.numpy())
    return mean_absolute_error(trues, preds)


def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    best_val_mae     = float("inf")
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            criterion(model(X_batch), y_batch).backward()
            optimizer.step()

        val_mae = _compute_mae(model, test_loader)
        scheduler.step(val_mae)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | train MAE: {_compute_mae(model, train_loader):.2f} | val MAE: {val_mae:.2f}")

        if val_mae < best_val_mae:
            best_val_mae     = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}  (best val MAE: {best_val_mae:.2f}분)")
                break

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))


def evaluate(model: nn.Module, loader: DataLoader, label: str) -> dict:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds.extend(np.maximum(0.0, model(X_batch).numpy()))
            trues.extend(y_batch.numpy())
    preds = np.array(preds)
    trues = np.array(trues)
    mae  = mean_absolute_error(trues, preds)
    rmse = float(np.sqrt(mean_squared_error(trues, preds)))
    print(f"[{label}] MAE: {mae:.2f}분 | RMSE: {rmse:.2f}분 | n={len(trues)}")
    return {"mae": mae, "rmse": rmse, "preds": preds, "trues": trues}


def evaluate_by_stage(preds: np.ndarray, trues: np.ndarray, elapsed: np.ndarray) -> None:
    print("\n=== 구간별 MAE (TEST) ===")
    for label, lo, hi in [("초반  0~30분", 0, 30), ("중반 31~90분", 31, 90), ("후반  91분+", 91, 9999)]:
        mask = (elapsed >= lo) & (elapsed <= hi)
        if mask.sum() == 0:
            continue
        mae = mean_absolute_error(trues[mask], preds[mask])
        print(f"  {label}: MAE {mae:.2f}분  (n={mask.sum()})")


def main() -> None:
    assert_pipeline_integrity(FEATURES)

    df_train_raw = assign_cycle_id(load_lstm_data(TRAIN_MONTHS))
    df_test_raw  = assign_cycle_id(load_lstm_data(TEST_MONTHS), id_offset=df_train_raw["cycle_id"].max())

    raw_elapsed_train = df_train_raw["elapsed_time"].copy()
    raw_elapsed_test  = df_test_raw["elapsed_time"].copy()

    df_train = df_train_raw.copy()
    df_test  = df_test_raw.copy()

    scaler = StandardScaler()
    df_train[FEATURES] = scaler.fit_transform(df_train[FEATURES])
    df_test[FEATURES]  = scaler.transform(df_test[FEATURES])

    X_train, y_train, elapsed_train = build_sequences(df_train, raw_elapsed_train)
    X_test,  y_test,  elapsed_test  = build_sequences(df_test,  raw_elapsed_test)

    print("=== 시퀀스 생성 결과 ===")
    print(f"  Train: {len(X_train):,}개 | Test: {len(X_test):,}개 | Total: {len(X_train)+len(X_test):,}개")
    print(f"  입력 shape: {X_train.shape}")
    print(f"\nT_remaining 분포 (train):\n{pd.Series(y_train).describe().round(2).to_string()}\n")

    train_loader = make_loader(X_train, y_train, shuffle=True)
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)

    model = CycleRemainingLSTM(input_size=len(FEATURES))
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}\n")

    train_model(model, train_loader, test_loader)

    print("\n=== 평가 ===")
    evaluate(model, train_loader, "TRAIN")
    result = evaluate(model, test_loader,  "TEST ")
    evaluate_by_stage(result["preds"], result["trues"], elapsed_test)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n모델 저장:    {MODEL_PATH}")
    print(f"스케일러 저장: {SCALER_PATH}")
    print(f"목표 대비: MAE {'PASS' if result['mae'] < 30 else 'FAIL'} (<30분) | "
          f"RMSE {'PASS' if result['rmse'] < 45 else 'FAIL'} (<45분)")


if __name__ == "__main__":
    main()
