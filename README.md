# 하수처리 포기기 가동 최적화 AI 시스템

XGBoost + LSTM 통합 모델 기반 **최적 포기기 가동 시간(cycle_runtime)** 예측 시스템

포기기를 너무 일찍 끄면 수질 처리 불량, 너무 늦게 끄면 전력 낭비 — AI가 수질 센서 데이터를 보고 적절한 종료 시점을 실시간으로 알려줍니다.

---

## 모델 구조

```
사이클 시작 → XGBoost: 수질 상태 보고 총 가동 시간 예측
10분마다    → LSTM:     실시간 수질 변화 보고 잔여 시간 보정
경과 60분+  → 두 모델 통합: max(XGBoost 잔여, LSTM 예측) 채택
T_signal ≤ 20분 → 포기기 중단
```

---

## 설치

```bash
pip install -r requirements.txt
```

### DB 연동 (선택)

PostgreSQL을 사용하는 경우 프로젝트 루트에 `.env` 파일 생성:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=aeration
DB_USER=postgres
DB_PASSWORD=yourpassword
```

`DB_HOST` 미설정 시 DB 없이 실행됩니다 (예측 기능은 정상 동작, DB 저장만 비활성화).

---

## 실행 방법

### 1. 모델 학습 (최초 1회)

```bash
py xgboost_pipeline.py    # XGBoost 학습 → models/ 저장
py lstm_pipeline.py       # LSTM 학습    → models/ 저장
py integration.py         # 통합 검증 + models/report.md 자동 생성
```

### 2. API 서버 시작

```bash
py -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### 3. 모니터링 대시보드

`dashboard.html` 을 브라우저로 열기

### 4. 시뮬레이션 테스트 (선택)

12월 테스트 데이터에서 XGBoost 예측 기준 가장 긴 사이클을 선택해 시뮬레이션합니다.

```bash
py test_cycle.py                          # 반응조A 단독 (기본값)
py test_cycle.py --reactor B              # 반응조B 단독
py test_cycle.py --reactor both           # A+B 병렬 동시 실행
py test_cycle.py --reactor both --interval 0.5  # 빠른 시뮬레이션 (0.5초 간격)
```

`both` 모드 실행 시 두 반응조가 threading으로 동시에 시뮬레이션되며, 대시보드에서 A/B 탭 전환으로 각 예측 추이를 확인할 수 있습니다.

### 5. 춘양 교차 검증

봉화 학습 모델을 미학습 처리장(춘양)에 적용해 일반화 성능을 검증합니다.

```bash
# 1단계: 춘양 원본 데이터 전처리
py preprocessing/preprocess_chunyang.py

# 2단계: 봉화 vs 춘양 교차 검증 실행
py validate_chunyang.py
```

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/cycle/start` | 사이클 시작 — XGBoost가 총 가동 시간 예측 |
| POST | `/cycle/predict` | 10분마다 호출 — 잔여 시간 신호(T_signal) 수신 |
| GET | `/cycle/history/{reactor}` | 현재 사이클 예측 이력 |
| GET | `/metrics` | 모델 성능 지표 + 전력 절감 + 운영 KPI |
| GET | `/health` | 서버 상태 확인 |

전체 API 문서: `http://localhost:8000/docs`

---

## 파일 구조

```
├── data/                           # 전처리된 학습/테스트 데이터
│   ├── 봉화_*월_xgb.csv            # 봉화 XGBoost 피처 (6~12월)
│   ├── 봉화_*월_lstm.csv           # 봉화 LSTM 피처 (6~12월)
│   ├── 춘양_*월_xgb.csv            # 춘양 XGBoost 피처 (6~12월, 9월 제외)
│   ├── 춘양_*월_lstm.csv           # 춘양 LSTM 피처 (6~12월, 9월 제외)
│   └── reference/
│       └── 경상북도_공공하수처리시설_현황.xlsx
├── models/                         # 학습된 모델 및 메타데이터
│   ├── xgboost_model.json          # XGBoost 모델
│   ├── lstm_model.pt               # LSTM 모델
│   ├── lstm_scaler.pkl             # LSTM 피처 스케일러
│   ├── feature_importance.csv      # XGBoost gain/weight (학습 후 자동 생성)
│   ├── shap_importance.csv         # SHAP 피처 중요도 (학습 후 자동 생성)
│   ├── metadata.json               # 학습 이력, 하이퍼파라미터, 피처 스냅샷
│   ├── report.md                   # 성능 리포트 (통합 검증 후 자동 생성)
│   └── archive/                    # 이전 버전 모델 보관
├── preprocessing/
│   ├── preprocessing.py            # 봉화 원본 CSV 전처리
│   └── preprocess_chunyang.py      # 춘양 원본 CSV 전처리
├── features.py                     # XGBoost/LSTM 피처 상수 (단일 진실 공급원)
├── utils.py                        # 공통 유틸리티 (parse_datetime, underprediction_rate, generate_report)
├── xgboost_pipeline.py             # XGBoost 학습
├── lstm_pipeline.py                # LSTM 학습
├── integration.py                  # 두 모델 통합 + 봉화 검증
├── validate_chunyang.py            # 춘양 교차 처리장 검증
├── server.py                       # FastAPI 서버
├── dashboard.html                  # 모니터링 대시보드
├── test_cycle.py                   # 시뮬레이션 테스트
└── CHANGES.md                      # 코드 개선 이력
```

---

## 모델 성능 (12월 테스트셋)

### 봉화 (학습 처리장)

| 구간 | MAE | 비고 |
|------|-----|------|
| 0~30분 | 39.32분 | XGBoost 단독 |
| 31~90분 | 55.15분 | 전환 구간 |
| 91분+ | 23.61분 | XGBoost + LSTM 통합 ✅ |

가동 종료 판단이 이루어지는 후반 구간(91분+)에서 MAE 23.61분 달성 (목표: <30분)

### 춘양 교차 검증 (미학습 처리장)

봉화에서 학습한 모델을 별도 전처리 없이 춘양 데이터에 직접 적용해 일반화 성능을 확인합니다.
결과는 `py validate_chunyang.py` 실행 후 콘솔 출력으로 확인할 수 있습니다.

---

## 운영 KPI

예측 정확도(MAE/RMSE) 외에 실제 운영 관점 지표를 측정합니다.

| KPI | 의미 |
|-----|------|
| 평균 절감 시간 | 사이클당 실제 가동 절감 시간 (에너지 비용) |
| 조기 종료율 | 실제 종료 전 STOP 신호 발생 비율 (처리 안전성) |
| 평균 초과 가동 | 늦게 끈 사이클의 평균 초과 시간 (에너지 낭비) |
| STOP 미발생율 | 사이클 내 STOP 신호가 한 번도 발생하지 않은 비율 |
| NH4 위험 사이클 | STOP 시점 NH4 > 5.0 mg/L 비율 (수질 안전성) |

---

## 전력 절감 추정

봉화읍 하수처리장 연간 전력 사용량(958,862 kWh) 기준, 포기기 비중 55% 적용

| 항목 | 값 |
|------|-----|
| 사이클당 평균 절감 | 8.2분 |
| 연간 절감 전력 | 62,118 kWh |
| 연간 절감 전기료 | 6,832,944원 |

> 시뮬레이션 기반 실제 STOP 시점 추정치 (재학습 후 `models/report.md`에 자동 갱신)

---

## 모델 안전성

| 항목 | 내용 |
|------|------|
| Leakage 방지 | XGBoost/LSTM 각각 FORBIDDEN 피처 assert — 학습 진입 전 차단 |
| Cycle boundary | groupby(cycle_id) 기반 diff/rolling — 사이클 간 데이터 오염 없음 |
| Scaler drift | transform 후 ±5σ clip — 센서 이상치로 인한 LSTM hidden state 폭발 방지 |
| 입력 검증 | Pydantic Field ge/le — 물리적으로 불가능한 센서값 422로 차단 |
| 재현성 | XGBoost `random_state=42`, PyTorch `manual_seed(42)` 고정 |
