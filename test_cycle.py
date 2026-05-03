import time
import requests
import pandas as pd
from pathlib import Path
from integration import parse_datetime

API      = "http://localhost:8000"
DATA_DIR = Path(__file__).parent / "data"
ENCODING = "cp949"
INTERVAL = 3


def load_longest_cycle():
    df_xgb = pd.read_csv(DATA_DIR / "봉화_12월_xgb.csv", encoding=ENCODING)
    df_xgb["수집시간"] = parse_datetime(df_xgb["수집시간"])

    df_lstm = pd.read_csv(DATA_DIR / "봉화_12월_lstm.csv", encoding=ENCODING)
    df_lstm["수집시간"] = parse_datetime(df_lstm["수집시간"])
    df_lstm = df_lstm[df_lstm["current_aeration"] == 1].copy().reset_index(drop=True)
    df_lstm["cycle_id"] = (df_lstm["elapsed_time"] == 0).cumsum()

    xgb_starts = df_xgb[df_xgb["elapsed_time"] == 0].set_index(["수집시간", "계측기명"])

    best_cycle, best_xgb_row = None, None
    for cid, cycle in df_lstm.groupby("cycle_id"):
        start_row = cycle.iloc[0]
        key = (start_row["수집시간"], start_row["계측기명"])
        if key not in xgb_starts.index:
            continue
        if best_cycle is None or len(cycle) > len(best_cycle):
            xgb_row = xgb_starts.loc[key]
            if isinstance(xgb_row, pd.DataFrame):
                xgb_row = xgb_row.iloc[0]
            best_cycle, best_xgb_row = cycle, xgb_row

    print(f"선택된 사이클: {len(best_cycle)}스텝 (최대 경과 {best_cycle['elapsed_time'].max()}분)")
    return best_xgb_row, best_cycle


def to_start_payload(row, reactor="반응조A"):
    return {
        "reactor":          reactor,
        "nh4":              float(row["nh4"]),
        "no3":              float(row["no3"]),
        "ph":               float(row["ph"]),
        "temp":             float(row["temp"]),
        "current_r":        float(row["상전류(R)"]),
        "nh4_no3_ratio":    float(row["nh4_no3_ratio"]),
        "do_saturation":    float(row["do_saturation"]),
        "nh4_diff":         float(row["nh4_diff"]),
        "no3_diff":         float(row["no3_diff"]),
        "ph_diff":          float(row["ph_diff"]),
        "nh4_rolling_mean": float(row["nh4_rolling_mean"]),
        "nh4_decay_rate":   float(row["nh4_decay_rate"]),
        "hour_sin":         float(row["hour_sin"]),
        "hour_cos":         float(row["hour_cos"]),
        "weekday":          int(row["weekday"]),
    }


def to_predict_payload(row, reactor="반응조A"):
    return {
        "reactor":          reactor,
        "elapsed_time":     int(row["elapsed_time"]),
        "nh4":              float(row["nh4"]),
        "no3":              float(row["no3"]),
        "ph":               float(row["ph"]),
        "temp":             float(row["temp"]),
        "current_r":        float(row["상전류(R)"]),
        "nh4_no3_ratio":    float(row["nh4_no3_ratio"]),
        "do_saturation":    float(row["do_saturation"]),
        "nh4_diff":         float(row["nh4_diff"]),
        "no3_diff":         float(row["no3_diff"]),
        "ph_diff":          float(row["ph_diff"]),
        "nh4_rolling_mean": float(row["nh4_rolling_mean"]),
        "nh4_decay_rate":   float(row["nh4_decay_rate"]),
        "hour_sin":         float(row["hour_sin"]),
        "hour_cos":         float(row["hour_cos"]),
        "weekday":          int(row["weekday"]),
    }


def main():
    print("데이터 로드 중...")
    xgb_row, cycle = load_longest_cycle()
    reactor = "반응조A"

    print(f"\n[1] /cycle/start 호출 → {reactor}")
    resp = requests.post(f"{API}/cycle/start", json=to_start_payload(xgb_row, reactor))
    resp.raise_for_status()
    result = resp.json()
    print(f"    XGBoost 예측 총 가동 시간: {result['cycle_runtime_xgb']}분\n")

    print(f"[2] /cycle/predict 순차 호출 ({INTERVAL}초 간격, 실제 10분 시뮬레이션)")
    print(f"    대시보드(dashboard.html) 열고 새로고침하면 그래프가 채워집니다.\n")
    print(f"    {'경과':>6}  {'T_signal':>8}  {'예측종료':>8}  {'모드'}")
    print(f"    {'-'*45}")

    for _, row in cycle.iterrows():
        payload = to_predict_payload(row, reactor)
        resp    = requests.post(f"{API}/cycle/predict", json=payload)
        resp.raise_for_status()
        r = resp.json()

        predicted_end = r["elapsed_time"] + r["t_signal"]
        print(f"    {r['elapsed_time']:>5}분  {r['t_signal']:>7}분  {predicted_end:>7}분  {r['mode']}")

        if not r["should_continue"]:
            print("\n    → should_continue: false → 폭기 중단 신호")
            break

        time.sleep(INTERVAL)

    print("\n완료. 대시보드에서 최종 그래프를 확인하세요.")


if __name__ == "__main__":
    main()
