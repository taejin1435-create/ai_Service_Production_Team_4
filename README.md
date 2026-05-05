# 하수처리 포기기 가동 최적화 AI 시스템

XGBoost + LSTM 통합 모델 기반 **최적 포기기 가동 시간(cycle_runtime)** 예측 시스템

포기기를 너무 일찍 끄면 수질 처리 불량, 너무 늦게 끄면 전력 낭비 — AI가 수질 센서 데이터를 보고 적절한 종료 시점을 실시간으로 알려줍니다.

---

## 모델 구조

```
사이클 시작 → XGBoost: 수질 상태 보고 총 가동 시간 예측
10분마다    → LSTM:     실시간 수질 변화 보고 잔여 시간 보정
경과 80분+  → 두 모델 통합: max(XGBoost 잔여, LSTM 예측) 채택
T_signal ≤ 10분 → 포기기 중단
```

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 실행 방법

### 1. API 서버 시작

```bash
py -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### 2. 모니터링 대시보드

`dashboard.html` 을 브라우저로 열기

### 3. 시뮬레이션 테스트 (선택)

12월 테스트 데이터에서 XGBoost 예측 기준 가장 긴 사이클을 선택해 시뮬레이션합니다.

```bash
py test_cycle.py                        # 반응조A 단독 (기본값)
py test_cycle.py --reactor B            # 반응조B 단독
py test_cycle.py --reactor both         # A+B 병렬 동시 실행
py test_cycle.py --reactor both --interval 1  # 빠른 시뮬레이션 (1초 간격)
```

`both` 모드 실행 시 두 반응조가 threading으로 동시에 시뮬레이션되며, 대시보드에서 A/B 탭 전환으로 각 예측 추이를 확인할 수 있습니다.

### 4. 춘양 교차 검증

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
| GET | `/metrics` | 모델 성능 지표 + 전력 절감 추정 |
| GET | `/health` | 서버 상태 확인 |

전체 API 문서: `http://localhost:8000/docs`

---

## 파일 구조

```
├── data/                           # 전처리된 학습/테스트 데이터
│   ├── 봉화_*월_xgb.csv            # 봉화 XGBoost 피처 (6~12월)
│   ├── 봉화_*월_lstm.csv           # 봉화 LSTM 피처 (6~12월)
│   ├── 춘양_*월_xgb.csv            # 춘양 XGBoost 피처 (6~12월)
│   ├── 춘양_*월_lstm.csv           # 춘양 LSTM 피처 (6~12월)
│   └── reference/
│       └── 경상북도_공공하수처리시설_현황.xlsx  # 봉화·춘양 처리장 기본 현황 참고자료
├── preprocessing/
│   ├── preprocessing.py            # 봉화 원본 CSV 전처리
│   └── preprocess_chunyang.py      # 춘양 원본 CSV 전처리
├── xgboost_pipeline.py             # XGBoost 학습
├── lstm_pipeline.py                # LSTM 학습
├── integration.py                  # 두 모델 통합 + 봉화 검증
├── validate_chunyang.py            # 춘양 교차 처리장 검증
├── server.py                       # FastAPI 서버
├── dashboard.html                  # 모니터링 대시보드
├── test_cycle.py                   # 시뮬레이션 테스트
├── xgboost_model.json              # 학습된 XGBoost 모델
├── lstm_model.pt                   # 학습된 LSTM 모델
└── lstm_scaler.pkl                 # LSTM 피처 스케일러
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

## 전력 절감 추정

봉화읍 하수처리장 연간 전력 사용량(958,862 kWh) 기준, 포기기 비중 55% 적용

| 항목 | 값 |
|------|-----|
| 사이클당 평균 절감 | 8.2분 |
| 연간 절감 전력 | 62,118 kWh |
| 연간 절감 전기료 | 6,832,944원 |

> 기존 운전 데이터 대비 AI 예측 시간 차이 기준 추정치
