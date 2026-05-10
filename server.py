import warnings
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from integration import AerationPredictor, LSTM_WINDOW_THRESHOLD, MIN_T_SIGNAL, validate_on_december

import uuid
import os
import logging
import threading
import json
import time
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from trainer import trainer
from dotenv import load_dotenv
load_dotenv()
# 로그 폴더 생성
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/aeration.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("aeration_api")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT"))
}

pool: SimpleConnectionPool = None

# DB 초기화
def init_db():
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS aeration_log (
            id SERIAL PRIMARY KEY,

            request_id TEXT,
            cycle_id TEXT,
            reactor TEXT,

            -- 🔹 원본 입력값
            nh4 REAL,
            no3 REAL,
            ph REAL,
            temp REAL,
            current_r REAL,

            nh4_no3_ratio REAL,
            do_saturation REAL,

            nh4_diff REAL,
            no3_diff REAL,
            ph_diff REAL,

            nh4_rolling_mean REAL,
            nh4_decay_rate REAL,

            hour_sin REAL,
            hour_cos REAL,
            weekday INTEGER,

            elapsed_time INTEGER,

            -- 🔹 모델 결과
            t_signal INTEGER,
            should_continue BOOLEAN,
            mode TEXT,

            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
    conn.commit()
    conn.close()

# DB 저장
def save_to_db(data: dict):
    def _task():
        conn = None

        try:
            conn = pool.getconn()

            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO aeration_log (
                    request_id, cycle_id, reactor,

                    nh4, no3, ph, temp, current_r,
                    nh4_no3_ratio, do_saturation,

                    nh4_diff, no3_diff, ph_diff,
                    nh4_rolling_mean, nh4_decay_rate,

                    hour_sin, hour_cos, weekday,
                    elapsed_time,

                    t_signal, should_continue, mode
                ) VALUES (
                    %s,%s,%s,
                    %s,%s,%s,%s,%s,
                    %s,%s,
                    %s,%s,%s,
                    %s,%s,
                    %s,%s,%s,
                    %s,
                    %s,%s,%s
                )
                """, (
                    data["request_id"],
                    data["cycle_id"],
                    data["reactor"],

                    data["nh4"],
                    data["no3"],
                    data["ph"],
                    data["temp"],
                    data["current_r"],

                    data["nh4_no3_ratio"],
                    data["do_saturation"],

                    data["nh4_diff"],
                    data["no3_diff"],
                    data["ph_diff"],

                    data["nh4_rolling_mean"],
                    data["nh4_decay_rate"],

                    data["hour_sin"],
                    data["hour_cos"],
                    data["weekday"],

                    data["elapsed_time"],

                    data["t_signal"],
                    data["should_continue"],
                    data["mode"]
                ))

            conn.commit()

            logger.info(
                f"[DB 저장 완료] cycle={data['cycle_id']} t={data['t_signal']}"
            )

        except Exception as e:
            logger.error(f"[DB 저장 실패] {str(e)}")

        finally:
            if conn:
                pool.putconn(conn)

    threading.Thread(target=_task, daemon=True).start()



warnings.filterwarnings("ignore")

VALID_REACTORS = {"반응조A", "반응조B"}
REACTOR_CODE   = {"반응조A": 0, "반응조B": 1}

predictors: dict[str, AerationPredictor] = {}
_metrics_cache: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    pool = SimpleConnectionPool(1, 10, **DB_CONFIG)
    init_db()
    for reactor in VALID_REACTORS:
        predictors[reactor] = AerationPredictor()
    print("모델 로드 완료 (반응조A, 반응조B)")
    try:
        validation_predictor = AerationPredictor()
        result = validate_on_december(validation_predictor)
        _metrics_cache.update(result)
    except Exception as e:
        print(f"검증 데이터 없음 (메트릭 비활성화): {e}")
    yield
    predictors.clear()
    _metrics_cache.clear()


app = FastAPI(
    title="하수처리 폭기 제어 API",
    description=(
        "XGBoost + LSTM 통합 모델 기반 **최적 폭기 가동 시간(cycle_runtime)** 예측 시스템\n\n"
        "### 사용 흐름\n"
        "1. 사이클 시작 시 `/cycle/start` 호출 → XGBoost가 총 가동 시간 예측\n"
        "2. 10분마다 `/cycle/predict` 호출 → 잔여 시간 신호(T_signal) 수신\n"
        "3. `should_continue: false` 수신 시 폭기 중단\n\n"
        "### 모델 구조\n"
        "- **경과 < 80분**: XGBoost 단독 예측\n"
        "- **경과 ≥ 80분**: XGBoost + LSTM 통합 (더 긴 시간 채택)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic 모델 ─────────────────────────────────────────────────────────────

class CycleStartRequest(BaseModel):
    reactor:          str   = Field(..., description="반응조 이름 (반응조A 또는 반응조B)")
    nh4:              float = Field(..., description="암모니아성 질소 농도 (mg/L)")
    no3:              float = Field(..., description="질산성 질소 농도 (mg/L)")
    ph:               float = Field(..., description="수소이온농도")
    temp:             float = Field(..., description="수온 (°C)")
    current_r:        float = Field(..., description="포기기 R상 전류 (A)")
    nh4_no3_ratio:    float = Field(..., description="NH4/NO3 비율")
    do_saturation:    float = Field(..., description="용존산소 포화도")
    nh4_diff:         float = Field(..., description="NH4 변화량 (직전 대비)")
    no3_diff:         float = Field(..., description="NO3 변화량 (직전 대비)")
    ph_diff:          float = Field(..., description="pH 변화량 (직전 대비)")
    nh4_rolling_mean: float = Field(..., description="NH4 이동평균 (최근 6스텝)")
    nh4_decay_rate:   float = Field(..., description="NH4 감소율")
    hour_sin:         float = Field(..., description="시간 sin 인코딩")
    hour_cos:         float = Field(..., description="시간 cos 인코딩")
    weekday:          int   = Field(..., description="요일 (0=월요일 ~ 6=일요일)")


class CycleStartResponse(BaseModel):
    reactor:           str = Field(..., description="반응조 이름")
    cycle_runtime_xgb: int = Field(..., description="XGBoost 예측 총 가동 시간 (분)")
    cycle_id:        str = Field(..., description="사이클 ID (사이클 시작 시 생성된 고유값)")

class PredictRequest(BaseModel):
    reactor:          str   = Field(..., description="반응조 이름 (반응조A 또는 반응조B)")
    elapsed_time:     int   = Field(..., description="사이클 시작 후 경과 시간 (분, 10분 단위)")
    nh4:              float = Field(..., description="암모니아성 질소 농도 (mg/L)")
    no3:              float = Field(..., description="질산성 질소 농도 (mg/L)")
    ph:               float = Field(..., description="수소이온농도")
    temp:             float = Field(..., description="수온 (°C)")
    current_r:        float = Field(..., description="포기기 R상 전류 (A)")
    nh4_no3_ratio:    float = Field(..., description="NH4/NO3 비율")
    do_saturation:    float = Field(..., description="용존산소 포화도")
    nh4_diff:         float = Field(..., description="NH4 변화량 (직전 대비)")
    no3_diff:         float = Field(..., description="NO3 변화량 (직전 대비)")
    ph_diff:          float = Field(..., description="pH 변화량 (직전 대비)")
    nh4_rolling_mean: float = Field(..., description="NH4 이동평균 (최근 6스텝)")
    nh4_decay_rate:   float = Field(..., description="NH4 감소율")
    hour_sin:         float = Field(..., description="시간 sin 인코딩")
    hour_cos:         float = Field(..., description="시간 cos 인코딩")
    weekday:          int   = Field(..., description="요일 (0=월요일 ~ 6=일요일)")
    cycle_id:        str   = Field(..., description="사이클 ID (사이클 시작 시 생성된 고유값)")

class PredictResponse(BaseModel):
    reactor:         str  = Field(..., description="반응조 이름")
    elapsed_time:    int  = Field(..., description="경과 시간 (분)")
    t_signal:        int  = Field(..., description="잔여 가동 시간 신호 (분)")
    mode:            str  = Field(..., description="예측 모드: xgb_only(경과<80분) 또는 integrated(경과≥80분)")
    should_continue: bool = Field(..., description="폭기 계속 여부 (false이면 중단)")


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _to_xgb_series(req: CycleStartRequest) -> pd.Series:
    return pd.Series({
        "nh4":              req.nh4,
        "no3":              req.no3,
        "ph":               req.ph,
        "temp":             req.temp,
        "상전류(R)":         req.current_r,
        "nh4_no3_ratio":    req.nh4_no3_ratio,
        "do_saturation":    req.do_saturation,
        "nh4_diff":         req.nh4_diff,
        "no3_diff":         req.no3_diff,
        "ph_diff":          req.ph_diff,
        "nh4_rolling_mean": req.nh4_rolling_mean,
        "nh4_decay_rate":   req.nh4_decay_rate,
        "hour_sin":         req.hour_sin,
        "hour_cos":         req.hour_cos,
        "weekday":          req.weekday,
        "reactor":          REACTOR_CODE[req.reactor],
    })


def _to_lstm_series(req: PredictRequest) -> pd.Series:
    return pd.Series({
        "nh4":              req.nh4,
        "no3":              req.no3,
        "ph":               req.ph,
        "temp":             req.temp,
        "상전류(R)":         req.current_r,
        "nh4_no3_ratio":    req.nh4_no3_ratio,
        "do_saturation":    req.do_saturation,
        "nh4_diff":         req.nh4_diff,
        "no3_diff":         req.no3_diff,
        "ph_diff":          req.ph_diff,
        "nh4_rolling_mean": req.nh4_rolling_mean,
        "nh4_decay_rate":   req.nh4_decay_rate,
        "hour_sin":         req.hour_sin,
        "hour_cos":         req.hour_cos,
        "weekday":          req.weekday,
        "elapsed_time":     req.elapsed_time,
    })


def _validate_reactor(reactor: str) -> None:
    if reactor not in VALID_REACTORS:
        raise HTTPException(
            status_code=422,
            detail=f"유효하지 않은 반응조입니다: '{reactor}'. 허용값: {sorted(VALID_REACTORS)}",
        )


# ── 엔드포인트 ─────────────────────────────────────────────────────────────────

@app.get(
    "/cycle/history/{reactor}",
    summary="현재 사이클 예측 이력",
    description="해당 반응조의 현재 사이클에서 호출된 예측값 이력을 반환합니다.",
    tags=["폭기 제어"],
)
async def cycle_history(reactor: str):
    _validate_reactor(reactor)
    return predictors[reactor].get_history()

@app.get(
    "/metrics",
    summary="모델 성능 지표",
    description="12월 테스트셋 기준 MAE, RMSE 등 통합 모델 평가 결과를 반환합니다.",
    tags=["시스템"],
)
async def get_metrics():
    if not _metrics_cache:
        raise HTTPException(status_code=503, detail="검증 데이터가 없어 메트릭을 계산할 수 없습니다.")
    return {
        "test_month": "12월",
        "overall": {
            "mae":  round(_metrics_cache["mae"],  2),
            "rmse": round(_metrics_cache["rmse"], 2),
        },
        "integrated": {
            "mae":       _metrics_cache.get("stage_91_mae",  0),
            "rmse":      _metrics_cache.get("stage_91_rmse", 0),
            "mae_pass":  _metrics_cache.get("stage_91_mae",  99) < 30,
            "rmse_pass": _metrics_cache.get("stage_91_rmse", 99) < 45,
        },
        "targets": {"mae": 30, "rmse": 45},
        "stages": _metrics_cache.get("stages", []),
        "model": {
            "xgb_threshold_min": LSTM_WINDOW_THRESHOLD,
            "window_size": 9,
            "hidden_size": 128,
            "integration": "max(xgb_remain, lstm_remain)",
        },
        "energy": _metrics_cache.get("energy"),
    }


@app.get(
    "/health",
    summary="서버 상태 확인",
    description="모델 로드 상태 및 서버 정상 동작 여부를 확인합니다.",
    tags=["시스템"],
)
async def health():
    return {
        "status": "정상",
        "loaded_reactors": list(predictors.keys()),
    }


@app.post(
    "/cycle/start",
    response_model=CycleStartResponse,
    summary="사이클 시작",
    description=(
        "폭기 사이클 시작 시점에 호출합니다.\n\n"
        "사이클 시작 시점의 수질 데이터를 입력하면 **XGBoost**가 "
        "이번 사이클의 **총 가동 시간(cycle_runtime)** 을 예측합니다.\n\n"
        "- 반드시 사이클 **시작 시점**(`elapsed_time == 0`)에 한 번만 호출\n"
        "- 이후 10분마다 `/cycle/predict` 호출로 잔여 시간을 수신"
    ),
    tags=["폭기 제어"],
)
async def cycle_start(req: CycleStartRequest):
    _validate_reactor(req.reactor)
    xgb_row = _to_xgb_series(req)
    predictor = predictors[req.reactor]
    cycle_runtime_xgb = predictors[req.reactor].on_cycle_start(xgb_row)
    cycle_id = str(uuid.uuid4())
    predictor.current_cycle_id = cycle_id
    logger.info(
        f"[START] reactor={req.reactor} runtime={cycle_runtime_xgb} cycle_id={cycle_id}"
    )
    return CycleStartResponse(
        reactor=req.reactor,
        cycle_runtime_xgb=cycle_runtime_xgb,
        cycle_id=cycle_id
    )


@app.post(
    "/cycle/predict",
    response_model=PredictResponse,
    summary="잔여 시간 예측 (10분마다 호출)",
    description=(
        "폭기 사이클 진행 중 **10분마다** 호출합니다.\n\n"
        "경과 시간에 따라 두 가지 예측 모드로 동작합니다:\n\n"
        "| 경과 시간 | 모드 | 설명 |\n"
        "|-----------|------|------|\n"
        "| < 80분 | `xgb_only` | XGBoost 단독: `cycle_runtime - elapsed_time` |\n"
        "| ≥ 80분 | `integrated` | XGBoost + LSTM 통합: 두 예측 중 더 긴 값 채택 |\n\n"
        "`should_continue: false` 수신 시 즉시 폭기를 중단하세요."
    ),
    tags=["폭기 제어"],
)
async def cycle_predict(req: PredictRequest):
    _validate_reactor(req.reactor)
    predictor  = predictors[req.reactor]
    cycle_id = req.cycle_id or getattr(predictor, "current_cycle_id", None)
    if cycle_id is None:
        raise HTTPException(400, "cycle_id가 없습니다. /cycle/start 먼저 호출하세요")
    lstm_row   = _to_lstm_series(req)
    t_signal   = predictor.predict(lstm_row, req.elapsed_time)
    cont       = predictor.should_continue(t_signal)
    mode       = "xgb_only" if req.elapsed_time < LSTM_WINDOW_THRESHOLD else "integrated"
    data = {
        "request_id": str(uuid.uuid4()),
        "cycle_id": cycle_id,
        "reactor": req.reactor,
        "nh4": req.nh4,
        "no3": req.no3,
        "ph": req.ph,
        "temp": req.temp,
        "current_r": req.current_r,

        "nh4_no3_ratio": req.nh4_no3_ratio,
        "do_saturation": req.do_saturation,

        "nh4_diff": req.nh4_diff,
        "no3_diff": req.no3_diff,
        "ph_diff": req.ph_diff,

        "nh4_rolling_mean": req.nh4_rolling_mean,
        "nh4_decay_rate": req.nh4_decay_rate,

        "hour_sin": req.hour_sin,
        "hour_cos": req.hour_cos,
        "weekday": req.weekday,

        "elapsed_time": req.elapsed_time,
        "t_signal": t_signal,
        "should_continue": cont,
        "mode": mode
    }

    save_to_db(data)

    logger.info(
        f"[PREDICT] reactor={req.reactor} "
        f"t_signal={t_signal} continue={cont} cycle_id={req.cycle_id}"
    )

    return PredictResponse(
        reactor=req.reactor,
        elapsed_time=req.elapsed_time,
        t_signal=t_signal,
        mode=mode,
        should_continue=cont,
    )
'''
@app.post(
    "/train",
    summary="모델 수동 재학습",
    description="DB 데이터를 기반으로 XGBoost 모델을 백그라운드 재학습합니다.",
    tags=["학습"]
)
async def train_model():

    result = trainer.train_async()

    return result
'''