import os
import logging
import uuid
import threading
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

import psycopg2
from psycopg2.pool import SimpleConnectionPool

from dotenv import load_dotenv
from cachetools import TTLCache

from integration import AerationPredictor, LSTM_WINDOW_THRESHOLD

# -------------------------------
# ENV
# -------------------------------
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "aeration_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "1234"),
    "port": int(os.getenv("DB_PORT", 5432))
}

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost").split(",")
FALLBACK_FILE = "fallback_log.jsonl"

# -------------------------------
# LOG
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aeration_api")

# -------------------------------
# DB POOL
# -------------------------------
pool: SimpleConnectionPool = None

class PooledConn:
    def __init__(self, conn, from_pool=True):
        self.conn = conn
        self.from_pool = from_pool

def get_conn_safe():
    try:
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return PooledConn(conn, True)
        except Exception:
            try:
                pool.putconn(conn, close=True)
            except Exception:
                pass
            return PooledConn(psycopg2.connect(**DB_CONFIG), False)
    except Exception:
        return PooledConn(psycopg2.connect(**DB_CONFIG), False)

def release_conn(pconn: PooledConn):
    try:
        if pconn.from_pool:
            pool.putconn(pconn.conn)
        else:
            pconn.conn.close()
    except Exception:
        pass

# -------------------------------
# DB INIT
# -------------------------------
def init_db():
    pconn = get_conn_safe()
    try:
        with pconn.conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS aeration_log (
                id SERIAL PRIMARY KEY,
                request_id TEXT,
                cycle_id TEXT,
                reactor TEXT,
                nh4 REAL,
                no3 REAL,
                ph REAL,
                temp REAL,
                elapsed_time INTEGER,
                t_signal INTEGER,
                should_continue BOOLEAN,
                mode TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
        pconn.conn.commit()
    finally:
        release_conn(pconn)

# -------------------------------
# FALLBACK
# -------------------------------
fallback_lock = threading.Lock()

def save_to_fallback(data: dict):
    try:
        with fallback_lock:
            with open(FALLBACK_FILE, "a") as f:
                f.write(json.dumps(data) + "\n")
    except Exception:
        logger.critical(f"Fallback 저장 실패 - 데이터 유실: {data.get('request_id')}")

# -------------------------------
# DB SAVE (비동기)
# -------------------------------
def save_to_db(data: dict):
    def _task():
        pconn = None
        try:
            pconn = get_conn_safe()
            conn = pconn.conn

            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO aeration_log (
                    request_id, cycle_id, reactor,
                    nh4, no3, ph, temp,
                    elapsed_time, t_signal,
                    should_continue, mode
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    data["request_id"], data["cycle_id"], data["reactor"],
                    data["nh4"], data["no3"], data["ph"], data["temp"],
                    data["elapsed_time"], data["t_signal"],
                    data["should_continue"], data["mode"]
                ))

            conn.commit()

        except Exception:
            save_to_fallback(data)

        finally:
            if pconn:
                release_conn(pconn)

    threading.Thread(target=_task, daemon=True).start()

# -------------------------------
# STATE
# -------------------------------
GLOBAL_PREDICTOR = AerationPredictor()
class CycleState:
    def __init__(self):
        self.predictor = GLOBAL_PREDICTOR
        self.last_used = datetime.now()
        self.lock = threading.Lock()

cycles: Dict[str, CycleState] = {}
cycles_lock = threading.Lock()

# -------------------------------
# CLEANUP THREAD
# -------------------------------
def cleanup_worker():
    while True:
        time.sleep(3600)
        now = datetime.now()
        with cycles_lock:
            expired = [
                cid for cid, c in cycles.items()
                if now - c.last_used > timedelta(hours=6)
            ]
            for cid in expired:
                del cycles[cid]

# -------------------------------
# IDEMPOTENCY
# -------------------------------
recent_requests = TTLCache(maxsize=10000, ttl=300)
recent_lock = threading.Lock()

def generate_key(req):
    data = req.model_dump()
    normalized = {
        k: round(v, 6) if isinstance(v, float) else v
        for k, v in data.items()
    }
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True).encode()
    ).hexdigest()

# -------------------------------
# LIFESPAN
# -------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    pool = SimpleConnectionPool(1, 10, **DB_CONFIG)

    init_db()

    threading.Thread(target=cleanup_worker, daemon=True).start()

    yield

    pool.closeall()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# MODEL
# -------------------------------
VALID_REACTORS = {"반응조A", "반응조B"}

class SensorInput(BaseModel):
    reactor: str
    nh4: float
    no3: float
    ph: float
    temp: float
    current_r: float
    nh4_no3_ratio: float
    do_saturation: float
    nh4_diff: float
    no3_diff: float
    ph_diff: float
    nh4_rolling_mean: float
    nh4_decay_rate: float
    hour_sin: float
    hour_cos: float
    weekday: int

    @field_validator("ph")
    def ph_check(cls, v):
        if not (0 <= v <= 14):
            raise ValueError("Invalid input")
        return v

    @field_validator("temp")
    def temp_check(cls, v):
        if not (-10 <= v <= 60):
            raise ValueError("Invalid input")
        return v

class PredictRequest(SensorInput):
    cycle_id: str
    elapsed_time: int

    @field_validator("elapsed_time")
    def elapsed_check(cls, v):
        if v < 0:
            raise ValueError("Invalid input")
        return v

# -------------------------------
# VALIDATION
# -------------------------------
def _check(req):
    if req.reactor not in VALID_REACTORS:
        raise HTTPException(422, "Invalid input")
    if abs(req.nh4) > 100 or abs(req.no3) > 100:
        raise HTTPException(400, "Invalid input")
REACTOR_CODE = {"반응조A": 0, "반응조B": 1}

def _to_series(req):
    data = req.model_dump()

    data["상전류(R)"] = data.pop("current_r")
    if hasattr(req, "cycle_id") is False:
        data["reactor"] = REACTOR_CODE.get(req.reactor, 0)

    return pd.Series(data)

# -------------------------------
# API
# -------------------------------
@app.post("/cycle/start")
def start(req: SensorInput):
    _check(req)

    key = generate_key(req)

    with recent_lock:
        entry = recent_requests.get(key)

        if entry:
            if entry["status"] == "READY":
                return {"cycle_id": entry["cycle_id"]}
            else:
                raise HTTPException(
                    status_code=409,
                    detail="Processing",
                    headers={"Retry-After": "2"}
                )

        cycle_id = str(uuid.uuid4())
        recent_requests[key] = {"status": "PENDING", "cycle_id": cycle_id}

    try:
        state = CycleState()
        runtime = state.predictor.on_cycle_start(_to_series(req))

        with cycles_lock:
            cycles[cycle_id] = state

        with recent_lock:
            recent_requests[key] = {"status": "READY", "cycle_id": cycle_id}

        return {"cycle_id": cycle_id, "runtime": runtime}

    except Exception:
        with recent_lock:
            recent_requests.pop(key, None)
        raise HTTPException(500, "Internal error")


@app.post("/cycle/predict")
def predict(req: PredictRequest):
    _check(req)

    with cycles_lock:
        state = cycles.get(req.cycle_id)

    if not state:
        raise HTTPException(404, "Invalid input")

    with state.lock:
        state.last_used = datetime.now()
        t_signal = state.predictor.predict(_to_series(req), req.elapsed_time)
        cont = state.predictor.should_continue(req.elapsed_time)

    mode = "xgb_only" if req.elapsed_time < LSTM_WINDOW_THRESHOLD else "integrated"

    request_id = str(uuid.uuid4())

    save_to_db({
        "request_id": request_id,
        "cycle_id": req.cycle_id,
        "reactor": req.reactor,
        "nh4": req.nh4,
        "no3": req.no3,
        "ph": req.ph,
        "temp": req.temp,
        "elapsed_time": req.elapsed_time,
        "t_signal": t_signal,
        "should_continue": cont,
        "mode": mode
    })

    return {
        "request_id": request_id,
        "t_signal": t_signal,
        "should_continue": cont,
        "mode": mode,
        "timestamp": datetime.now().isoformat()
    }


@app.delete("/cycle/{cycle_id}")
def delete_cycle(cycle_id: str):
    deleted = False

    with cycles_lock:
        if cycle_id in cycles:
            del cycles[cycle_id]
            deleted = True

    with recent_lock:
        keys_to_delete = [
            k for k, v in recent_requests.items()
            if v["cycle_id"] == cycle_id
        ]
        for k in keys_to_delete:
            del recent_requests[k]

    if not deleted:
        raise HTTPException(404, "Not found")

    return {"message": "deleted"}