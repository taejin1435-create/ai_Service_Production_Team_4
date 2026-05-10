# 코드 개선 이력

파일별 변경 이력. 변경 번호(#N)는 전체 작업 순서 기준.

---

## `features.py` _(신규)_

### #2 Feature Constants 중앙화

기존에 `preprocessing.py`, `xgboost_pipeline.py`, `lstm_pipeline.py`, `integration.py` 4곳에 feature 리스트가 분산되어 train/inference mismatch 위험 존재. 단일 진실 공급원(single source of truth)으로 통합.

```python
XGB_FEATURES              # XGBoost 입력 feature (16개)
XGB_MONOTONE_CONSTRAINTS  # XGB feature와 1:1 대응 단조성 제약
XGB_TARGET                # "cycle_runtime"
XGB_FORBIDDEN             # {"elapsed_time", "T_remaining", "current_aeration", "cycle_runtime"}

LSTM_FEATURES             # LSTM 입력 feature (16개)
LSTM_TARGET               # "T_remaining"
LSTM_FORBIDDEN            # {"cycle_runtime", "current_aeration", "T_remaining"}
```

**발견된 기존 버그:** `preprocessing.py`의 LSTM_FEATURES가 실제 학습 리스트(16개)가 아닌 12개짜리였음. 중앙화로 수정.

### #12 XGBoost Target Leakage Assert 강화

`XGB_FORBIDDEN`에 `cycle_runtime`(XGB 타겟 자체)이 누락되어 있었음. `assert_pipeline_integrity()`가 타겟을 feature로 넣는 치명적 leakage를 감지하지 못하는 상태였음.

```python
# 이전
XGB_FORBIDDEN = {"elapsed_time", "T_remaining", "current_aeration"}

# 이후
XGB_FORBIDDEN = {"elapsed_time", "T_remaining", "current_aeration", "cycle_runtime"}
```

---

## `utils.py`

### #3 Underprediction Safety Metric 추가

MAE/RMSE만으로는 과대/과소 방향성 구분 불가. 포기기를 너무 일찍 끄는 것(과소예측)이 실제 운영의 핵심 위험이라 안전 metric 추가.

```python
def underprediction_rate(preds, trues, threshold=20):
    return float((preds < (trues - threshold)).mean())
```

### #5 `parse_datetime` 공용 utils 통합

`parse_datetime`이 4개 파일에 동일하게 복제되어 있었음 (`integration.py`, `lstm_pipeline.py`, `xgboost_pipeline.py`, `preprocessing.py`). 단일 정의로 통합.

```python
def parse_datetime(series: pd.Series) -> pd.Series:
    result = pd.to_datetime(series, format="%m월%d월%y %H:%M", errors="coerce")
    mask = result.isna()
    if mask.any():
        result[mask] = pd.to_datetime(series[mask], errors="coerce")
    return result
```

**사이드 노트:** `"%m월%d월%y %H:%M"` 포맷은 코드 버그가 아니라 9월 xgb CSV가 실제로 사용하는 한국식 포맷. 나머지 월은 ISO 포맷이라 fallback 분기로 처리됨.

### #27 `generate_report` 키 이름 불일치 버그 수정

`stop.get('early_stop_rate')` → `stop.get('unsafe_early_stop_rate')`. `update_metadata`가 `unsafe_early_stop_rate` 키로 저장하는데 report 생성 시 다른 키로 조회하여 "조기 종료율" 항목이 항상 "—"로 출력되던 문제 수정.

### #17 `update_metadata` / `generate_report` 신규 함수

```
xgboost_pipeline.main() → update_metadata(xgb_mae, xgb_rmse, trained_at)
lstm_pipeline.main()    → update_metadata(lstm_mae, lstm_rmse)
integration.validate_on_december() → update_metadata(통합 metrics) → generate_report()
```

`report.md` 포함 섹션: Data / XGBoost / LSTM / Integrated / Top SHAP Features / Operational STOP KPIs / Model Integrity Checks

---

## `preprocessing/preprocessing.py`

### #1 Cycle Boundary Leakage 제거

`diff()`와 `rolling()`이 전역 계산되어 이전 cycle 마지막 값과 다음 cycle 첫 값이 연결되는 문제 수정. LSTM sequence 오염, fake gradient, 잘못된 decay signal 발생 가능성 제거.

| feature | 이전 | 이후 |
|---|---|---|
| `nh4_diff` | `df["nh4"].diff()` | `df.groupby(cycle_id)["nh4"].diff()` |
| `no3_diff` | `df["no3"].diff()` | `df.groupby(cycle_id)["no3"].diff()` |
| `ph_diff` | `df["ph"].diff()` | `df.groupby(cycle_id)["ph"].diff()` |
| `nh4_decay_rate` | `nh4_diff / 10` | `groupby(cycle_id)["nh4"].diff(3) / 30` |
| `nh4_rolling_mean` | `df["nh4"].rolling(6)` | `groupby(cycle_id)["nh4"].transform(rolling(6))` |

`nh4_decay_rate`는 단순 파생값에서 독립적인 30분 단위 평균 기울기로 변경. `preprocess_chunyang.py`는 `preprocessing.py`를 import하므로 자동 반영.

### #2 Feature Constants 중앙화 반영

로컬 `LSTM_FEATURES`, `LSTM_FORBIDDEN_FEATURES` 정의 제거 후 `from features import ...`로 대체.

---

## `xgboost_pipeline.py`

### #2 Feature Constants 중앙화 반영

로컬 `FEATURES`, `MONOTONE_CONSTRAINTS`, `TARGET`, `FORBIDDEN_FEATURES` 정의 제거 후 import.

### #4 학습 데이터 확장 (7~11월 + 9월 imputation)

기존 `MONTHS = [8, 9, 10, 12]`에서 9월은 `temp` / `do_saturation` 100% NaN으로 `dropna` 단계에서 전멸. 실효 학습 month는 [8, 10]뿐이었음.

```python
# 이전
MONTHS = [8, 9, 10, 12]

# 이후
MONTHS = [7, 8, 9, 10, 11, 12]

SEPT_IMPUTE = {
    "temp":          (23.87, 21.33),
    "do_saturation": (8.437, 8.843),
}
```

| 월 | 이전 | 이후 |
|---|---|---|
| 7월 | 미사용 | 701 |
| 9월 | 0 (dropna 전멸) | 637 |
| 11월 | 미사용 | 619 |
| **TRAIN 합계** | ~1,273 | **3,230 (약 2.5배)** |

### #15 Feature Importance / SHAP CSV 자동 저장

```python
df.to_csv(MODELS_DIR / "feature_importance.csv")
df.to_csv(MODELS_DIR / "shap_importance.csv")
```

### #17 metadata 자동 업데이트 반영

`main()` 종료 전 `update_metadata(xgb_mae, xgb_rmse, trained_at)` 호출 추가.

### #19 학습 데이터 6월 추가 및 하이퍼파라미터 조정

```python
# 이전
MONTHS = [7, 8, 9, 10, 11, 12]
min_child_weight = 10

# 이후
MONTHS = [6, 7, 8, 9, 10, 11, 12]
min_child_weight = 5   # 더 세밀한 트리 분할 허용
```

---

## `lstm_pipeline.py`

### #2 Feature Constants 중앙화 반영

로컬 `FEATURES`, `TARGET`, `FORBIDDEN` 정의 제거 후 import.

### #14 Random Seed 고정

```python
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

import 직후, 경로 상수 이전에 배치. `DataLoader(shuffle=True)`의 epoch별 sample 순서 및 LSTM weight 초기화 재현성 확보.

### #17 metadata 자동 업데이트 반영

`main()` 종료 전 `update_metadata(lstm_mae, lstm_rmse)` 호출 추가.

### #19 학습 데이터 6월 추가 및 하이퍼파라미터 조정

```python
# 이전
TRAIN_MONTHS = [7, 8, 9, 10, 11]
WINDOW_SIZE  = 9
HUBER_DELTA  = 5.0
MAX_EPOCHS   = 100
PATIENCE     = 15

# 이후
TRAIN_MONTHS = [6, 7, 8, 9, 10, 11]
WINDOW_SIZE  = 7      # LSTM 활성화 80분 → 60분 (평균 사이클 76.7분 대비 80분은 실효성 낮음)
HUBER_DELTA  = 10.0   # T_remaining 범위(10~230분) 대비 delta=5가 너무 작았음
MAX_EPOCHS   = 150
PATIENCE     = 20
```

---

## `integration.py`

### #2 Feature Constants 중앙화 반영

로컬 `XGB_FEATURES`, `LSTM_FEATURES` 정의 제거 후 import.

### #6 수질 안전 Metric 추가 — NH4 불량 위험 사이클 비율

데이터 기반 threshold 도출: 봉화 전월 LSTM 데이터 4,275개 사이클 종료 시점 NH4 95th percentile = 5.773 mg/L.

```python
NH4_SAFETY_THRESHOLD = 5.0  # 신규 상수

# cycle 루프 내부
if stop_nh4 is None and t_signal <= STOP_THRESHOLD:
    stop_nh4 = float(row["nh4"])

# 집계
nh4_risk_rate = (stop_nh4_arr > NH4_SAFETY_THRESHOLD).mean()
```

### #7 xgb_remain 음수 방지

```python
# 이전
xgb_remain = self._cycle_runtime_xgb - raw_elapsed

# 이후
xgb_remain = max(0, self._cycle_runtime_xgb - raw_elapsed)
```

`np.clip` 이전 단계의 신호 정합성 보장. `max(xgb_remain, lstm_remain)` 계산 전 음수 개입 차단.

### #8 Predictor Buffer Elapsed-Based Reset

`on_cycle_start()` 누락 시 이전 사이클 끝 버퍼 위에 새 사이클 데이터가 누적되는 문제 방지. `elapsed_time` 감소 감지 방식 채택.

```python
self._last_elapsed: int = -1  # __init__ 및 on_cycle_start에 추가

# predict 진입부 (현재 최종)
if raw_elapsed < self._last_elapsed:   # < 로 변경 (#22에서 수정)
    self._buffer.clear()
# raw_elapsed == self._last_elapsed: duplicate packet, 무시
self._last_elapsed = raw_elapsed
```

### #9 Scaler Drift Protection — Scaled Feature Clipping

운영 환경 센서 이상치 유입 시 LSTM hidden state 폭발 방지. transform 후 scaled 값 기준 ±5σ clipping.

```python
scaled = self.scaler.transform(pd.DataFrame([lstm_row[LSTM_FEATURES]]))[0].astype(np.float32)
scaled = np.clip(scaled, -5.0, 5.0)
self._buffer.append(scaled)
```

### #10 cycle_runtime 최소값 하한 — MIN_T_SIGNAL 기준 통일

```python
# 이전
self._cycle_runtime_xgb = int(max(0, pred))

# 이후
self._cycle_runtime_xgb = int(max(MIN_T_SIGNAL, pred))
```

### #11 모델 디렉토리 구조화

```python
MODELS_DIR      = BASE_DIR / "models"
XGB_MODEL_PATH  = MODELS_DIR / "xgboost_model.json"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.pt"
SCALER_PATH     = MODELS_DIR / "lstm_scaler.pkl"
```

### #17 metadata 자동 업데이트 / report.md 생성 반영

`validate_on_december()` 종료 전 `update_metadata()` → `generate_report()` 호출.

**버그 수정:** 기존에 `return` 이후에 이 호출이 위치하여 실행되지 않았음.

### #18 운영 STOP 시뮬레이션 KPI

사이클 루프 내 `t_signal <= STOP_THRESHOLD` 첫 발생 시점의 `elapsed_time`을 `stop_elapsed`로 기록. 에너지 절감 계산을 XGBoost 초기 예측 기반 → 실제 STOP 시점 기반으로 교체.

| KPI | 계산 | 의미 |
|---|---|---|
| `avg_time_saved` | `mean(actual - stop_elapsed)` | 사이클당 평균 절감 시간 |
| `unsafe_early_stop_rate` | `(stopped_diff > EARLY_STOP_MARGIN).mean()` | 20분+ 조기 종료 위험 비율 |
| `avg_overrun` | `mean(max(0, stop - actual))` | 불필요 초과 가동 평균 |
| `no_stop_rate` | `(stop is None).mean()` | STOP 신호 미발생율 |
| `nh4_risk_rate` | `(nh4_stop > 5.0).mean()` | 수질 위반 위험 |

### #20 STOP_THRESHOLD 분리 및 통합 로직 실험

clip 하한과 STOP 조건이 동일값(10분)으로 묶여 STOP 미발생율 57.6%. 분리로 해소.

```python
# 이전
MIN_T_SIGNAL = 10  # clip 하한 겸 STOP 기준

# 이후
MIN_T_SIGNAL   = 10  # clip 하한
STOP_THRESHOLD = 20  # STOP 판단 기준
```

LSTM 100% 전환 실험 후 춘양 교차 검증 FAIL(STOP 미발생율 99.3%) 확인 → `max(xgb_remain, lstm_remain)` 원상복구.

| 지표 | 변경 전 | 변경 후 (봉화) | 춘양(미학습) |
|---|---|---|---|
| STOP 미발생율 | 57.6% | 42.8% | 99.3% |
| 평균 절감 시간 | -8.3분 | +4.0분 | -73.3분 |
| 91분+ MAE | — | 25.30분 ✅ | 64.39분 ❌ |

춘양 FAIL은 봉화 단일 처리장 학습 기반 일반화 한계로 결론.

### #21 LSTM 단독 MAE 하드코딩 제거

```python
# 이전
print(f"  (참고) LSTM 단독 MAE: 34.03분 / RMSE: 42.12분")

# 이후
meta      = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8")) if ... else {}
lstm_mae  = meta.get("metrics", {}).get("lstm_mae", "—")
lstm_rmse = meta.get("metrics", {}).get("lstm_rmse", "—")
```

`import json` 상단 이동. 함수 내부 지역변수명 언더스코어 prefix 제거.

### #22 운영 안정성 수정

```python
# ① clip 하한 MIN_T_SIGNAL 복구 (0 출력 방지)
t_signal = int(np.clip(t_signal, MIN_T_SIGNAL, MAX_T_SIGNAL))

# ② duplicate packet 무시 (< 로 변경)
if raw_elapsed < self._last_elapsed:
    self._buffer.clear()

# ③ EARLY_STOP_MARGIN 도입
EARLY_STOP_MARGIN = 20
unsafe_early_stop_rate = float((stopped_diff > EARLY_STOP_MARGIN).mean())

# ④ predicted_total_min dead code 제거

# ⑤ n_clamped — clip 전후 비교로 실제 clamping 감지
raw_t_signal = t_signal
t_signal = int(np.clip(t_signal, MIN_T_SIGNAL, MAX_T_SIGNAL))
was_clamped = raw_t_signal != t_signal

# ⑥ early_stop_rate → unsafe_early_stop_rate 리네임
```

### #23 `predict()` 반환 타입 변경 및 `_log` 개선

```python
# 반환 타입
def predict(self, lstm_row: pd.Series, raw_elapsed: int) -> tuple[int, bool]:
    ...
    return t_signal, was_clamped

# _log에 was_clamped 추가
self._log.append({
    "elapsed_time":       raw_elapsed,
    "t_signal":           t_signal,
    "predicted_end_time": raw_elapsed + t_signal,
    "xgb_prediction":     self._cycle_runtime_xgb,
    "mode":               mode,
    "was_clamped":        was_clamped,
})
```

### #28 `stage_91_mae` / `stage_91_rmse` 기본값 버그 수정

```python
# 이전 — 데이터 없을 때 0 반환 → 0 < 30 = True → 잘못된 PASS
stage_91_mae  = ... if mask_91.sum() > 0 else 0

# 이후
stage_91_mae  = ... if mask_91.sum() > 0 else None
stage_91_rmse = ... if mask_91.sum() > 0 else None
```

`main()`도 None 안전 처리:

```python
mae_status  = "PASS ✅" if mae_91  is not None and mae_91  < 30 else "FAIL ❌"
rmse_status = "PASS ✅" if rmse_91 is not None and rmse_91 < 45 else "FAIL ❌"
```

`validate_chunyang.py`는 이미 `None` 반환으로 일관성도 함께 확보.

### #24 `deque[np.ndarray]` 타입 어노테이션, `import json` 상단 이동

```python
self._buffer: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
```

### #25 `current_cycle_id` 정식 선언

dynamic attribute 방지. `__init__` 및 `on_cycle_start`에 선언.

```python
self.current_cycle_id: str | None = None  # __init__
self.current_cycle_id = None              # on_cycle_start (사이클 교체 시 초기화)
```

### #29 `/metrics` 엔드포인트 None 처리 버그 수정

`stage_91_mae` / `stage_91_rmse`가 `None`일 때 `.get(key, default)`가 `None`을 반환해 `None < 30` → `TypeError` 발생 가능.

```python
# 이전
"mae_pass": _metrics_cache.get("stage_91_mae", 99) < 30

# 이후
mae_91  = _metrics_cache.get("stage_91_mae")
rmse_91 = _metrics_cache.get("stage_91_rmse")
...
"mae_pass":  mae_91  is not None and mae_91  < 30,
"rmse_pass": rmse_91 is not None and rmse_91 < 45,
```

`integration.py #28`의 None 반환 변경과 연동된 수정.

### #30 `PredictResponse`에 `was_clamped` 필드 추가

```python
class PredictResponse(BaseModel):
    ...
    was_clamped: bool = Field(..., description="Safety Clamping 적용 여부")
```

`cycle_predict`에서 `t_signal, _ = predictor.predict(...)` → `t_signal, was_clamped = predictor.predict(...)` 로 변경 후 응답에 포함. `test_cycle.py` STOP 로그에서 clamping 여부 실시간 확인 가능.

---

## `validate_chunyang.py`

### #3, #5, #6, #17, #18 반영

`integration.py`에서 추가된 `underprediction_rate`, `NH4_SAFETY_THRESHOLD`, STOP KPI 로직을 동일하게 적용. `parse_datetime`도 `utils`에서 import.

### #20 STOP_THRESHOLD import 추가

```python
from integration import (
    AerationPredictor, LSTM_WINDOW_THRESHOLD,
    MIN_T_SIGNAL, STOP_THRESHOLD, EARLY_STOP_MARGIN,
    MAX_T_SIGNAL, NH4_SAFETY_THRESHOLD, validate_on_december,
)
```

### #22 unsafe_early_stop_rate 반영

```python
unsafe_early_stop_rate = float((stopped_diff > EARLY_STOP_MARGIN).mean())
```

### #23 predict() 튜플 언패킹 반영

```python
t_signal, _ = predictor.predict(row, raw_elapsed)
```

---

## `server.py`

### #13 Pydantic Field 입력 범위 Validation

| field | 제약 | 근거 |
|---|---|---|
| `ph` | `ge=0, le=14` | 물리 법칙 |
| `temp` | `ge=0, le=50` | 하수처리 운영 범위 |
| `nh4`, `no3`, `nh4_rolling_mean`, `current_r`, `nh4_no3_ratio` | `ge=0` | 농도/전류 비음수 |
| `do_saturation` | `ge=0, le=200` | % 포화도 |
| `hour_sin`, `hour_cos` | `ge=-1, le=1` | sin/cos 정의역 |
| `weekday` | `ge=0, le=6` | 요일 인코딩 |
| `elapsed_time` | `ge=0, le=480` | 사이클 길이 상한 |

### #26 DB 저장 / logging / cycle_id 추가

- `logging` (파일 + 스트림 핸들러, `logs/aeration.log`)
- `cycle_id` UUID 생성 및 추적 (`/cycle/start` 응답, `/cycle/predict` 요청 필드)
- PostgreSQL DB 저장 (`psycopg2` + `SimpleConnectionPool` + daemon thread)
- `load_dotenv` + 환경변수 기반 `DB_CONFIG`
- `_DB_ENABLED` 플래그: `DB_HOST` 미설정 시 DB 없이 graceful 실행
- API 설명문 하드코딩 "80분" → `LSTM_WINDOW_THRESHOLD` 상수 참조
- `from integration import WINDOW_SIZE, HIDDEN_SIZE` (이중 관리 제거)
- `/health` 응답에 `db_enabled` 필드 추가
- `t_signal, _ = predictor.predict(...)` 튜플 언패킹
- `int(os.getenv("DB_PORT", "5432"))` 미설정 시 크래시 방지

---

## `test_cycle.py`

### #30 신규 파일 — 사이클 시뮬레이션 테스트

주요 구현:
- `cycle_id`를 `/cycle/start` 응답에서 추출해 `/cycle/predict` 요청에 전달
- 모델 경로: `BASE_DIR / "models" / "xgboost_model.json"`
- `resp.json()` → `result` 변수 재사용

구조 개선:
- `load_data()` 분리 — 모델/CSV `main()`에서 1회 로드. `both` 모드 XGBoost 2회 로드 제거
- `load_longest_cycle()` — reactor 필터링만 담당, 데이터 로드 분리
- 중복 `cycle_id` 계산 제거: `load_data()`에서 이미 생성된 `cycle_id` 재사용. `load_longest_cycle()` 내부 `lstm_reactor["cycle_id"] = ...` 삭제
- `requests.Session` — TCP 연결 재사용, `with` 블록으로 자동 해제
- `daemon=True` 제거 — `t.join()` 존재 시 불필요, 중간 종료 리스크 제거

안정성 강화:
- `_post()` 헬퍼 도입 — timeout, 예외 처리 중복 제거
- `timeout=REQUEST_TIMEOUT(10)` — 서버 hang 시 무한 대기 방지
- `requests.RequestException` — `ConnectionError`, `Timeout`, DNS 실패 포함 처리. `resp` 미생성 케이스도 `hasattr(e, "response")` 체크로 안전 처리
- `cycle_id` 빈 문자열 검증 — `RuntimeError` 발생
- `predicted_end = int(...) + int(...)` — 타입 명시로 string 혼입 방지

UX / 디버깅:
- `--interval` `type=int` → `type=float` (`--interval 0.5` 가능)
- STOP 로그: `elapsed`, `predicted_end`, `mode`, `was_clamped` 포함
- `load_longest_cycle()` 반환 타입 힌트 추가 `-> tuple[pd.Series, pd.DataFrame]`

---

## `models/metadata.json` _(신규)_

### #11 모델 디렉토리 구조화

```
models/
├── xgboost_model.json
├── lstm_model.pt
├── lstm_scaler.pkl
├── metadata.json
└── archive/
```

### #16 `model_version` 필드 추가

| version | 내용 |
|---|---|
| `v1.0.0` | baseline |
| `v1.1.0` | cycle boundary leakage fix |
| `v1.2.0` | XGBoost 학습 데이터 확장 (7~11월) |
| `v1.3.0` | sigma clipping + leakage assert + MIN_T_SIGNAL 하한 |

`features` 스냅샷 포함: 재학습 후 feature 리스트 변경 시 이전 scaler 로드로 인한 silent mismatch 방지.
