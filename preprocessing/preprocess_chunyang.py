import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import (
    load_and_merge,
    split_by_reactor,
    process_reactor,
    build_xgb_dataset,
    build_lstm_dataset,
    print_summary,
    ENCODING,
)
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).parent.parent
OLD_DATA = BASE_DIR / "old_data"
DATA_DIR = BASE_DIR / "data"

MONTHS = {
    "12월": ("춘양_12월_수질계측기.csv", "춘양_12월_온도진동전류.csv"),
    "11월": ("춘양_11월_수질계측기.csv", "춘양_11월_온도진동전류.csv"),
    "10월": ("춘양_10월_수질계측기.csv", "춘양_10월_온도전류전압.csv"),
    "8월":  ("춘양_8월_수질계측기.csv",  "춘양_8월_온도전류.csv"),
    "7월":  ("춘양_7월_수질계측기.csv",  "춘양_7월_온도전류.csv"),
    "6월":  ("춘양_6월_수질계측기.csv",  "춘양_6월_온도전류.csv"),
}

TARGET_MONTH = "6월"


def preprocess_month(month: str) -> tuple:
    wq_file, tc_file = MONTHS[month]
    quality_path = OLD_DATA / month / wq_file
    current_path = OLD_DATA / month / tc_file

    df = load_and_merge(quality_path, current_path)
    df_A, df_B = split_by_reactor(df)
    df_A = process_reactor(df_A)
    df_B = process_reactor(df_B)
    return df_A, df_B


if __name__ == "__main__":
    le = LabelEncoder()
    le.fit(["반응조A", "반응조B"])

    print(f"춘양 {TARGET_MONTH} 전처리 중...")
    df_A, df_B = preprocess_month(TARGET_MONTH)

    xgb_out  = DATA_DIR / f"춘양_{TARGET_MONTH}_xgb.csv"
    lstm_out = DATA_DIR / f"춘양_{TARGET_MONTH}_lstm.csv"

    xgb_df = build_xgb_dataset(df_A.copy(), df_B.copy(), le)
    xgb_df.to_csv(xgb_out, index=False, encoding=ENCODING)
    print_summary(f"춘양 {TARGET_MONTH} XGB", xgb_df)

    lstm_df = build_lstm_dataset(df_A.copy(), df_B.copy())
    lstm_df.to_csv(lstm_out, index=False, encoding=ENCODING)
    print_summary(f"춘양 {TARGET_MONTH} LSTM", lstm_df)

    print(f"\n저장 완료")
    print(f"  XGB  → {xgb_out}")
    print(f"  LSTM → {lstm_out}")
