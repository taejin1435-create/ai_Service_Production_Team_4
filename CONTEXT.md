# 프로젝트 컨텍스트 — 하수처리 포기기 가동 최적화 AI

> 새 채팅에서 이어서 작업할 때 이 파일을 먼저 읽어줘.

---

## 프로젝트 개요

**목표**: 하수처리장 포기기(폭기기)의 최적 가동 시간(`cycle_runtime`) 예측  
**장소**: 경상북도 봉화읍 하수처리장  
**모델**: XGBoost + LSTM 통합  
**데이터**: 수질 센서(NH4, NO3, pH, 수온) + 전류 센서(상전류 R)

### 작동 원리
```
사이클 시작 → XGBoost: 총 가동 시간 예측 (cycle_runtime)
10분마다    → LSTM: 잔여 시간(T_remaining) 실시간 보정
경과 80분+  → 통합 신호: max(XGBoost 잔여, LSTM 예측)
T_signal ≤ 10분 → 포기기 중단
```

---

## 파일 구조

```
wastewater-aeration-ai/
├── data/
│   ├── 봉화_7월_xgb.csv        # 7월 XGBoost 학습 데이터
│   ├── 봉화_7월_lstm.csv       # 7월 LSTM 학습 데이터
│   ├── 봉화_12월_xgb.csv       # 12월 테스트 데이터 (XGB용)
│   ├── 봉화_12월_lstm.csv      # 12월 테스트 데이터 (LSTM용)
│   └── reference/
│       └── 경상북도_공공하수처리시설_현황.xlsx  # 경상북도 전체 하수처리시설 현황 (전력, 수질 등)
├── old_data/                   # 원본 CSV (전처리 전)
│   └── 7월/
│       ├── 봉화_7월_수질계측기.csv
│       └── 봉화_7월_온도진동전류.csv
├── preprocessing/
│   └── preprocessing.py        # 원본 CSV 전처리
├── xgboost_pipeline.py         # XGBoost 학습
├── lstm_pipeline.py            # LSTM 학습
├── integration.py              # 두 모델 통합 + 12월 검증
├── server.py                   # FastAPI 서버
├── dashboard.html              # 모니터링 대시보드 (React + Chart.js)
├── test_cycle.py               # 시뮬레이션 테스트
├── xgboost_model.json          # 학습된 XGBoost 모델
├── lstm_model.pt               # 학습된 LSTM 모델
├── lstm_scaler.pkl             # LSTM 피처 스케일러
├── requirements.txt
├── README.md
└── CLAUDE.md                   # 코드 작성 원칙
```

---

## 모델 성능 (12월 테스트셋)

### 봉화 (학습 데이터)
| 구간 | 샘플 수 | MAE | 비고 |
|------|---------|-----|------|
| 0~30분 | 2,095 | 39.32분 | XGBoost 단독, FAIL |
| 31~90분 | 1,627 | 55.15분 | 전환 구간, FAIL |
| **91분+** | **1,201** | **23.61분** | **XGB+LSTM 통합, PASS ✅** |

- **헤드라인 지표**: 91분+ 구간 MAE (종료 판단 구간) — 23.61분 PASS
- 목표: MAE < 30분, RMSE < 45분

### 춘양 교차 처리장 검증 (미학습 — 봉화 모델 그대로 적용)
| 구간 | 샘플 수 | MAE | 비고 |
|------|---------|-----|------|
| 0~30분 | 1,231 | 28.25분 | XGBoost 단독 |
| 31~90분 | 1,606 | 20.51분 | 전환 구간 |
| **91분+** | **489** | **29.35분** | **XGB+LSTM 통합, PASS ✅** |
| 전체 | 3,326 | 24.67분 | RMSE 36.55분 |

- 봉화 학습 모델이 한 번도 본 적 없는 춘양 처리장에서도 목표 기준 통과
- 춘양이 전체 MAE 더 낮은 이유: 규모(800㎥/일 vs 3,000㎥/일) 차이로 사이클 패턴이 더 단순
- **결론: 모델 일반화 능력 확인 — 다른 처리장 추가 학습 없이 바로 적용 가능성 입증**

---

## 주요 상수 (integration.py)

```python
WINDOW_SIZE           = 9
LSTM_WINDOW_THRESHOLD = 80   # 80분 이후부터 LSTM 개입
MIN_T_SIGNAL          = 10   # 최소 잔여 시간 신호 (분)
MAX_T_SIGNAL          = 250  # 최대 잔여 시간 신호 (분)

# 전력 절감 계산용 (봉화읍 2023 실적)
AERATOR_KW_PER_MIN = 958_862 * 0.55 / 365 / 1440  # ≈ 1.003 kWh/분
KRW_PER_KWH        = 110
DECEMBER_DAYS      = 31
```

---

## API 엔드포인트 (server.py)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/cycle/start` | 사이클 시작, XGBoost 예측 반환 |
| POST | `/cycle/predict` | 10분마다 호출, T_signal 반환 |
| GET | `/cycle/history/{reactor}` | 현재 사이클 예측 이력 |
| GET | `/metrics` | 모델 성능 + 전력 절감 추정 |
| GET | `/health` | 서버 상태 |

### should_continue 로직 (server.py)
```python
# T_signal 기반으로 판단 (XGBoost 기준 아님)
cont = t_signal > MIN_T_SIGNAL
```
> ⚠️ 이전에는 `predictor.should_continue(raw_elapsed)`로 XGBoost 기준만 봤는데,
> LSTM이 연장을 판단해도 강제 종료되는 버그가 있었음. 수정 완료.

---

## /metrics 응답 구조

```json
{
  "test_month": "12월",
  "overall":    { "mae": 40.72, "rmse": 58.88 },
  "integrated": {
    "mae": 23.61, "rmse": 32.87,
    "mae_pass": true, "rmse_pass": true
  },
  "targets": { "mae": 30, "rmse": 45 },
  "stages": [
    { "label": "0~30분",  "mae": 39.32, "n": 2095 },
    { "label": "31~90분", "mae": 55.15, "n": 1627 },
    { "label": "91분+",   "mae": 23.61, "n": 1201 }
  ],
  "model": { "xgb_threshold_min": 80, ... },
  "energy": {
    "source": "봉화읍 하수처리장 (경상북도 2023)",
    "annual_total_kwh": 958862,
    "aerator_ratio": 0.55,
    "kwh_per_min": 1.0034,
    "krw_per_kwh": 110,
    "saved_kwh_annual": 62118,
    "saved_krw_annual": 6832944,
    "saved_min_per_cycle": 8.2
  }
}
```

---

## dashboard.html 구조

React (createElement, no JSX) + Chart.js 4.4.1 (cdnjs CDN)

### 컴포넌트 목록
- `HealthCard` — 서버 상태
- `MetricCard` — MAE/RMSE (종료 판단 구간 91분+ 기준)
- `CycleLineChart` — 실시간 예측 추이 (파란선: 예측종료, 노란점선: XGBoost 초기값)
- `StageBarChart` — 구간별 MAE 막대그래프 (목표 30분 기준선 표시)
- `StageTable` — 구간별 상세 테이블
- `EnergySavingsCard` — 연간 전력 절감 추정
- `ModelCard` — 모델 구조 정보

### Chart.js 주의사항
- 두 개의 `useEffect` 분리 필수: 마운트 시 생성 / 데이터 변경 시 업데이트
- canvas를 항상 DOM에 유지 (조건부 렌더링 금지 → 차트 destroy/recreate 루프 발생)
- 30초마다 history만 자동 폴링 (전체 새로고침 아님)

---

## test_cycle.py 주요 사항

### load_longest_cycle() 동작
1. XGBoost 모델을 직접 로드
2. 반응조A 데이터만 필터링 (두 반응조 섞이면 elapsed_time 순서 뒤섞임)
3. XGBoost 예측값 >= 80분인 사이클만 후보로 선정
4. 그 중 LSTM 스텝이 가장 많은 사이클 선택

### 해결된 버그들
- **반응조 미분리**: df_lstm에 A+B 섞여서 elapsed_time이 0,20,10,30... 순서로 뒤섞임 → 반응조A만 필터링
- **짧은 사이클 선택**: XGBoost 예측 < 80분인 사이클 선택 시 LSTM 개입 없이 종료 → XGBoost 예측 >= 80분 필터
- **조기 종료**: should_continue가 XGBoost 기준만 봐서 LSTM이 연장 판단해도 종료 → T_signal > MIN_T_SIGNAL 기준으로 변경

---

## 실행 방법

```bash
# 1. 서버 시작
py -m uvicorn server:app --reload --host 0.0.0.0 --port 8000

# 2. 대시보드
# dashboard.html 브라우저로 열기

# 3. 시뮬레이션 (서버 실행 중에)
py test_cycle.py                   # 반응조A 단독 (기본값)
py test_cycle.py --reactor B       # 반응조B 단독
py test_cycle.py --reactor both    # A+B 병렬 동시 실행
py test_cycle.py --reactor both --interval 1   # 빠른 시뮬레이션
```

---

## 전력 절감 계산 근거

- 출처: 경상북도 하수처리장 현황 (봉화읍, 2023)
- 연간 총 전력: 958,862 kWh/년
- 포기기 비중: 55% (일반적인 하수처리장 기준)
- 분당 전력: 958,862 × 0.55 ÷ 365 ÷ 1440 ≈ 1.003 kWh/분
- 전기료: 110원/kWh (한국 산업용)
- 절감 추정: 12월 테스트 데이터 기준 × 연간 환산 (×365/31)
- **결과: 연간 62,118 kWh / 6,832,944원 절감 추정**
- ⚠️ "기존 운전 데이터 대비 AI 예측 시간 차이" 기준 — 추정치임을 명시 필요

---

## GitHub

- 팀 레포: `https://github.com/taejin1435-create/ai_Service_Production_Team_4`
- 브랜치: `main` (force push로 팀 레포 정리 완료)
- `.gitignore`: `__pycache__/`, `*.pyc`, `*.pyo`
- `CLAUDE.md`는 개인 작업 원칙 파일 — 공유 여부는 본인 판단

---

## 현재 상태 / 남은 작업

✅ 완료
- XGBoost + LSTM 학습 및 통합
- FastAPI 서버
- React 대시보드 (실시간 차트, 성능 지표, 전력 절감 카드, A/B 탭)
- test_cycle.py — `--reactor A|B|both` CLI 인자 + A/B 병렬 threading 지원
- GitHub 업로드
- 춘양 교차 처리장 검증 (봉화 모델 → 춘양 91분+ MAE 29.35분 PASS)
- preprocessing.py 날짜 형식 파싱 개선 (merge 전 개별 변환)

### 춘양 교차 검증 실행 순서
```bash
# 1. 춘양 전처리 (TARGET_MONTH 변수로 월 지정)
py preprocessing/preprocess_chunyang.py

# 2. 봉화(학습) vs 춘양(미학습) 비교 검증
py validate_chunyang.py
```

🔲 필요 시 추가 가능
- 1~5월 봉화 데이터 재학습 → **불가** (no3 컬럼 없음 + 상전류(R) 없음, 아래 참고)
- 실제 PLC/SCADA 연동 (현재는 API 기반 수동 호출)

### 1~5월 봉화 데이터 사용 불가 사유
| 문제 | 1~3월 | 4~5월 |
|------|-------|-------|
| `no3` 컬럼 없음 | ❌ | ❌ |
| `상전류(R)` 없음 | ❌ (온도/진동만 있음) | ❌ (온도값/진동값만 있음) |
| 파일 형식 | xlsx | xlsx |

- `no3` 없음 → 핵심 피처 결측, 보간 불가 수준
- `상전류(R)` 없음 → `current_aeration` 판단 불가 → `cycle_runtime`, `T_remaining` 생성 불가
- **결론: 전처리 파이프라인에 투입 불가. 6월(데이터 수집 체계 개편 시점)부터만 사용 가능**
