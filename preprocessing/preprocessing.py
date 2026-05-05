import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# ── 경로 설정 (이 부분만 수정하여 다른 달에 재사용) ──────────────────────────
_BASE            = Path(__file__).parent.parent
QUALITY_PATH     = _BASE / "old_data" / "7월" / "봉화_7월_수질계측기.csv"
CURRENT_PATH     = _BASE / "old_data" / "7월" / "봉화_7월_온도진동전류.csv"
XGB_OUTPUT_PATH  = _BASE / "data" / "봉화_7월_xgb.csv"
LSTM_OUTPUT_PATH = _BASE / "data" / "봉화_7월_lstm.csv"
# ─────────────────────────────────────────────────────────────────────────────

ENCODING = "cp949"
SENTINEL_VALUE = -999
AERATION_CURRENT_THRESHOLD = 3.0
NO3_ABSOLUTE_THRESHOLD = 20.0
NO3_SPIKE_WINDOW = 5
NO3_SPIKE_STD_THRESHOLD = 5.0

QUALITY_REACTORS = ["반응조 A", "반응조 B"]
CURRENT_REACTORS = ["MOP-202A 포기기", "MOP-202B 포기기"]
REACTOR_NAME_MAP = {"MOP-202A 포기기": "반응조A", "MOP-202B 포기기": "반응조B"}

SENTINEL_COLS = ["nh4", "ph"]
INTERP_COLS   = ["nh4", "no3", "ph"]

QUALITY_HEADER = ["계측기명", "수집시간", "nh4", "_unnamed", "no3", "ph", "temp",
                  "toc", "ss", "pump_remote", "pump_run", "pump_fault",
                  "valve_remote", "valve_open", "valve_close", "s_can_fail"]
CURRENT_HEADER = ["계측기이름", "수집시간", "온도", "진동", "상전류(R)",
                  "상전류(S)", "상전류(T)", "상전압(R)", "상전압(S)", "상전압(T)"]

LSTM_FORBIDDEN_FEATURES = {"T_remaining", "cycle_runtime"}
LSTM_FEATURES = [
    "nh4", "no3", "ph", "temp",
    "nh4_diff", "no3_diff", "ph_diff",
    "hour_sin", "hour_cos",
    "elapsed_time", "nh4_decay_rate", "상전류(R)",
]


def _parse_datetime(series: pd.Series) -> pd.Series:
    result = pd.to_datetime(series, format="%m월%d월%y %H:%M", errors="coerce")
    mask = result.isna()
    if mask.any():
        result[mask] = pd.to_datetime(series[mask], errors="coerce")
    return result


def _read_csv_auto(path: Path, expected_header_col: str, fallback_cols: list[str]) -> pd.DataFrame:
    peek = pd.read_csv(path, encoding=ENCODING, nrows=0)
    has_header = expected_header_col in peek.columns
    df = pd.read_csv(path, encoding=ENCODING, header=0 if has_header else None)
    if not has_header:
        df.columns = fallback_cols[:len(df.columns)]
    return df


def load_and_merge(quality_path: Path, current_path: Path) -> pd.DataFrame:
    quality = _read_csv_auto(quality_path, "계측기명", QUALITY_HEADER)
    current = _read_csv_auto(current_path, "계측기이름", CURRENT_HEADER)

    quality = quality[quality["계측기명"].isin(QUALITY_REACTORS)].copy()
    current = current[current["계측기이름"].isin(CURRENT_REACTORS)].copy()

    quality = quality[["계측기명", "수집시간", "nh4", "no3", "ph", "temp"]].copy()
    quality["계측기명"] = quality["계측기명"].str.replace(" ", "", regex=False)
    quality["수집시간"] = _parse_datetime(quality["수집시간"])

    current = current[["계측기이름", "수집시간", "상전류(R)"]].copy()
    current["계측기명"] = current["계측기이름"].map(REACTOR_NAME_MAP)
    current = current.drop(columns=["계측기이름"])
    current["수집시간"] = _parse_datetime(current["수집시간"])

    df = pd.merge(quality, current, on=["계측기명", "수집시간"], how="left")
    df = df.sort_values(["계측기명", "수집시간"]).reset_index(drop=True)
    return df


def split_by_reactor(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_A = df[df["계측기명"] == "반응조A"].copy().reset_index(drop=True)
    df_B = df[df["계측기명"] == "반응조B"].copy().reset_index(drop=True)
    return df_A, df_B


def replace_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    df[SENTINEL_COLS] = df[SENTINEL_COLS].replace(SENTINEL_VALUE, np.nan)
    return df


def remove_no3_spike(df: pd.DataFrame) -> pd.DataFrame:
    no3 = df["no3"].copy()
    rolling_median = no3.rolling(window=NO3_SPIKE_WINDOW, center=True, min_periods=1).median()
    rolling_std    = no3.rolling(window=NO3_SPIKE_WINDOW, center=True, min_periods=1).std().fillna(1)
    is_statistical_spike = (no3 - rolling_median).abs() > NO3_SPIKE_STD_THRESHOLD * rolling_std
    is_absolute_spike    = no3 > NO3_ABSOLUTE_THRESHOLD
    df.loc[is_statistical_spike | is_absolute_spike, "no3"] = np.nan
    return df


def interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    df[INTERP_COLS] = df[INTERP_COLS].interpolate(method="linear", limit_direction="both")
    return df


def add_current_aeration(df: pd.DataFrame) -> pd.DataFrame:
    df["current_aeration"] = (df["상전류(R)"] > AERATION_CURRENT_THRESHOLD).astype(int)
    return df


def add_cycle_runtime(df: pd.DataFrame) -> pd.DataFrame:
    aeration = df["current_aeration"]
    cycle_id = (aeration != aeration.shift(1)).cumsum()
    cycle_lengths = (
        df[aeration == 1]
        .groupby(cycle_id[aeration == 1])
        .size() * 10
    )
    df["cycle_runtime"] = cycle_id.map(cycle_lengths).fillna(0).astype(int)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    aeration = df["current_aeration"]
    cycle_id = (aeration != aeration.shift(1)).cumsum()

    df["elapsed_time"]   = df.groupby(cycle_id).cumcount() * 10 * aeration
    df["T_remaining"]    = df["cycle_runtime"] - df["elapsed_time"]

    df["nh4_diff"]       = df["nh4"].diff().fillna(0)
    df["no3_diff"]       = df["no3"].diff().fillna(0)
    df["ph_diff"]        = df["ph"].diff().fillna(0)
    df["nh4_decay_rate"] = df["nh4_diff"] / 10

    df["hour"]    = df["수집시간"].dt.hour
    df["weekday"] = df["수집시간"].dt.weekday
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["nh4_rolling_mean"] = df["nh4"].rolling(window=6, min_periods=1).mean()
    df["nh4_no3_ratio"]    = df["nh4"] / (df["no3"] + 1e-6)
    df["do_saturation"]    = 468 / (31.6 + df["temp"])

    return df


def add_reactor_encoding(df: pd.DataFrame, le: LabelEncoder) -> pd.DataFrame:
    df["reactor"] = le.transform(df["계측기명"])
    return df


def filter_xgb_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask_invalid = (df["cycle_runtime"] == 0) & (df["current_aeration"] == 1)
    df = df[~mask_invalid]
    df = df[df["current_aeration"] == 1]
    return df.reset_index(drop=True)


def validate_lstm_features(feature_cols: list[str]) -> None:
    leaked = LSTM_FORBIDDEN_FEATURES & set(feature_cols)
    if leaked:
        raise ValueError(f"[LEAKAGE DETECTED] LSTM 입력에 금지 컬럼 포함: {leaked}")


def process_reactor(df: pd.DataFrame) -> pd.DataFrame:
    df = replace_sentinel(df)
    df = remove_no3_spike(df)
    df = interpolate_missing(df)
    df = add_current_aeration(df)
    df = add_cycle_runtime(df)
    df = add_derived_features(df)
    return df


def print_summary(label: str, df: pd.DataFrame) -> None:
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    print(f"shape      : {df.shape}")
    print(f"반응조A     : {(df['계측기명']=='반응조A').sum()}행")
    print(f"반응조B     : {(df['계측기명']=='반응조B').sum()}행")
    print(f"결측값 합계 : {df.isnull().sum().sum()}")
    if "T_remaining" in df.columns:
        print(f"T_remaining 음수 : {(df['T_remaining'] < 0).sum()}")
    print(df.head(3).to_string())


def build_xgb_dataset(df_A: pd.DataFrame, df_B: pd.DataFrame, le: LabelEncoder) -> pd.DataFrame:
    processed = []
    for reactor_df in [df_A, df_B]:
        reactor_df = add_reactor_encoding(reactor_df, le)
        reactor_df = filter_xgb_training_rows(reactor_df)
        processed.append(reactor_df)

    return pd.concat(processed).sort_values(
        by=["계측기명", "수집시간"]
    ).reset_index(drop=True)


def build_lstm_dataset(df_A: pd.DataFrame, df_B: pd.DataFrame) -> pd.DataFrame:
    validate_lstm_features(LSTM_FEATURES)
    return pd.concat([df_A, df_B]).sort_values(
        by=["수집시간", "계측기명"]
    ).reset_index(drop=True)


if __name__ == "__main__":
    print("파일 로드 및 merge...")
    df = load_and_merge(QUALITY_PATH, CURRENT_PATH)

    print("반응조별 분리 및 전처리...")
    df_A, df_B = split_by_reactor(df)
    df_A = process_reactor(df_A)
    df_B = process_reactor(df_B)

    le = LabelEncoder()
    le.fit(["반응조A", "반응조B"])

    print("XGBoost 데이터셋 생성...")
    xgb_df = build_xgb_dataset(df_A.copy(), df_B.copy(), le)
    xgb_df.to_csv(XGB_OUTPUT_PATH, index=False, encoding="cp949")
    print_summary("XGB 최종 데이터셋", xgb_df)

    print("\nLSTM 데이터셋 생성...")
    lstm_df = build_lstm_dataset(df_A.copy(), df_B.copy())
    lstm_df.to_csv(LSTM_OUTPUT_PATH, index=False, encoding="cp949")
    print_summary("LSTM 최종 데이터셋", lstm_df)

    print(f"\n저장 완료")
    pri