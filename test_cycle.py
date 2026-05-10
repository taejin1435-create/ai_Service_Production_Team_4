import argparse
import threading
import time

import pandas as pd
import requests
import xgboost as xgb
from pathlib import Path

from integration import XGB_FEATURES, LSTM_WINDOW_THRESHOLD
from utils import parse_datetime

API             = "http://localhost:8000"
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
ENCODING        = "cp949"
INTERVAL        = 3.0
REQUEST_TIMEOUT = 10

_print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def load_data() -> tuple[xgb.XGBRegressor, pd.DataFrame, pd.DataFrame]:
    model = xgb.XGBRegressor()
    model.load_model(str(BASE_DIR / "models" / "xgboost_model.json"))

    df_xgb = pd.read_csv(DATA_DIR / "봉화_12월_xgb.csv", encoding=ENCODING)
    df_xgb["수집시간"] = parse_datetime(df_xgb["수집시간"])

    df_lstm = pd.read_csv(DATA_DIR / "봉화_12월_lstm.csv", encoding=ENCODING)
    df_lstm["수집시간"] = parse_datetime(df_lstm["수집시간"])
    df_lstm = df_lstm[df_lstm["current_aeration"] == 1].copy().reset_index(drop=True)
    df_lstm["cycle_id"] = (df_lstm["elapsed_time"] == 0).cumsum()

    return model, df_xgb, df_lstm


def load_longest_cycle(
    reactor: str,
    model: xgb.XGBRegressor,
    df_xgb: pd.DataFrame,
    df_lstm: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    xgb_reactor  = df_xgb[df_xgb["계측기명"] == reactor].copy().reset_index(drop=True)
    lstm_reactor = df_lstm[df_lstm["계측기명"] == reactor].copy().reset_index(drop=True)

    starts = xgb_reactor[xgb_reactor["elapsed_time"] == 0].copy()
    starts["xgb_pred"] = model.predict(starts[XGB_FEATURES]).astype(int)
    xgb_starts = (
        starts[starts["xgb_pred"] >= LSTM_WINDOW_THRESHOLD]
        .set_index(["수집시간", "계측기명"])
    )

    best_cycle, best_xgb_row = None, None
    for _, cycle in lstm_reactor.groupby("cycle_id"):
        start_row = cycle.iloc[0]
        key = (start_row["수집시간"], start_row["계측기명"])
        if key not in xgb_starts.index:
            continue
        if best_cycle is None or len(cycle) > len(best_cycle):
            xgb_row = xgb_starts.loc[key]
            if isinstance(xgb_row, pd.DataFrame):
                xgb_row = xgb_row.iloc[0]
            best_cycle, best_xgb_row = cycle, xgb_row

    if best_cycle is None:
        raise RuntimeError(f"{reactor}: LSTM >= {LSTM_WINDOW_THRESHOLD}분 사이클 없음")

    safe_print(
        f"[{reactor}] 선택 사이클: {len(best_cycle)}스텝 / "
        f"최대 경과 {best_cycle['elapsed_time'].max()}분 / "
        f"XGB 예측 {best_xgb_row['xgb_pred']}분"
    )
    return best_xgb_row, best_cycle


def to_start_payload(row, reactor: str) -> dict:
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


def to_predict_payload(row, reactor: str, cycle_id: str) -> dict:
    return {
        "reactor":          reactor,
        "cycle_id":         cycle_id,
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


def _post(session: requests.Session, url: str, payload: dict) -> dict:
    try:
        resp = session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        body = e.response.text if hasattr(e, "response") and e.response is not None else ""
        safe_print(f"[HTTP ERROR] {e}\n{body}")
        raise


def run_simulation(
    reactor: str,
    interval: float,
    model: xgb.XGBRegressor,
    df_xgb: pd.DataFrame,
    df_lstm: pd.DataFrame,
) -> None:
    xgb_row, cycle = load_longest_cycle(reactor, model, df_xgb, df_lstm)

    with requests.Session() as session:
        result      = _post(session, f"{API}/cycle/start", to_start_payload(xgb_row, reactor))
        xgb_runtime = result["cycle_runtime_xgb"]
        cycle_id    = result["cycle_id"]

        if not cycle_id:
            raise RuntimeError(f"{reactor}: cycle_id 없음 — /cycle/start 응답 확인 필요")

        safe_print(f"[{reactor}] /cycle/start → XGBoost 예측: {xgb_runtime}분 (cycle_id: {cycle_id})")
        safe_print(f"[{reactor}] {'경과':>6}  {'T_signal':>8}  {'예측종료':>8}  {'모드'}")
        safe_print(f"[{reactor}] {'-'*45}")

        for _, row in cycle.iterrows():
            r             = _post(session, f"{API}/cycle/predict", to_predict_payload(row, reactor, cycle_id))
            predicted_end = int(r["elapsed_time"]) + int(r["t_signal"])
            safe_print(f"[{reactor}] {r['elapsed_time']:>5}분  {r['t_signal']:>7}분  {predicted_end:>7}분  {r['mode']}")

            if not r["should_continue"]:
                safe_print(
                    f"[{reactor}] → STOP "
                    f"(elapsed={r['elapsed_time']}분, "
                    f"predicted_end={predicted_end}분, "
                    f"mode={r['mode']}, "
                    f"clamped={r['was_clamped']})"
                )
                break

            time.sleep(interval)

    safe_print(f"[{reactor}] 완료.")


def main():
    parser = argparse.ArgumentParser(description="하수처리 포기기 사이클 시뮬레이션")
    parser.add_argument(
        "--reactor",
        choices=["A", "B", "both"],
        default="A",
        help="시뮬레이션 대상 반응조 (기본값: A)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=INTERVAL,
        help=f"스텝 간 대기 시간 초 (기본값: {INTERVAL})",
    )
    args = parser.parse_args()

    reactor_map = {
        "A":    ["반응조A"],
        "B":    ["반응조B"],
        "both": ["반응조A", "반응조B"],
    }
    targets = reactor_map[args.reactor]

    print(f"시뮬레이션 시작 — {', '.join(targets)} / 스텝 간격: {args.interval}초")
    print("대시보드(dashboard.html)를 열고 새로고침하면 실시간 그래프가 채워집니다.\n")
    print("데이터 로드 중...")
    model, df_xgb, df_lstm = load_data()

    if len(targets) == 1:
        run_simulation(targets[0], args.interval, model, df_xgb, df_lstm)
    else:
        threads = [
            threading.Thread(target=run_simulation, args=(r, args.interval, model, df_xgb, df_lstm))
            for r in targets
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print("\n전체 완료. 대시보드에서 A/B 탭을 전환해 결과를 확인하세요.")


if __name__ == "__main__":
    main()
