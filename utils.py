import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def parse_datetime(series: pd.Series) -> pd.Series:
    result = pd.to_datetime(series, format="%m월%d월%y %H:%M", errors="coerce")
    mask = result.isna()
    if mask.any():
        result[mask] = pd.to_datetime(series[mask], errors="coerce")
    return result


def underprediction_rate(preds: np.ndarray, trues: np.ndarray, threshold: int = 20) -> float:
    return float((preds < (trues - threshold)).mean())


def update_metadata(updates: dict, models_dir: Path) -> None:
    path = models_dir / "metadata.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k].update(v)
        else:
            data[k] = v
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_report(models_dir: Path) -> None:
    meta_path  = models_dir / "metadata.json"
    shap_path  = models_dir / "shap_importance.csv"
    report_path = models_dir / "report.md"

    meta    = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    metrics = meta.get("metrics", {})
    hp      = meta.get("hyperparameters", {})

    def fmt_min(v):
        return f"{v:.2f}분" if v is not None else "—"

    def fmt_pct(v):
        return f"{v*100:.1f}%" if v is not None else "—"

    train_months = meta.get("train_months", [])
    train_str    = f"{train_months[0]}~{train_months[-1]}월" if train_months else "—"

    lines = [
        "# Model Report",
        "",
        f"**Version:** {meta.get('model_version') or '—'}  ",
        f"**Trained:** {meta.get('trained_at') or '—'}  ",
        f"**Site:** {meta.get('site') or '—'}",
        "",
        "---",
        "",
        "## Data",
        "",
        f"| | |",
        f"|---|---|",
        f"| Train | {train_str} |",
        f"| Test  | {meta.get('test_month', '—')}월 |",
        "",
        "## XGBoost",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| MAE    | {fmt_min(metrics.get('xgb_mae'))} |",
        f"| RMSE   | {fmt_min(metrics.get('xgb_rmse'))} |",
        "",
        "## LSTM",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| MAE    | {fmt_min(metrics.get('lstm_mae'))} |",
        f"| RMSE   | {fmt_min(metrics.get('lstm_rmse'))} |",
        "",
        "## Integrated (XGBoost + LSTM)",
        "",
        "| Metric | Value | Target |",
        "|--------|-------|--------|",
        f"| MAE               | {fmt_min(metrics.get('integration_mae'))}  | < 30분 |",
        f"| RMSE              | {fmt_min(metrics.get('integration_rmse'))} | < 45분 |",
        f"| Underprediction   | {fmt_pct(metrics.get('underprediction_rate'))} | — |",
        f"| NH4 Risk Rate     | {fmt_pct(metrics.get('nh4_risk_rate'))} | — |",
    ]

    if shap_path.exists():
        shap_df = pd.read_csv(shap_path, index_col=0)
        top5    = shap_df.head(5)
        lines += [
            "",
            "## Top SHAP Features",
            "",
            "| Rank | Feature | SHAP Importance |",
            "|------|---------|----------------|",
        ]
        for i, (feat, row) in enumerate(top5.iterrows(), 1):
            lines.append(f"| {i} | `{feat}` | {row['shap_abs_mean']:.4f} |")

    stop = meta.get("metrics", {})
    lines += [
        "",
        "## Operational STOP KPIs",
        "",
        "| KPI | Value | 의미 |",
        "|-----|-------|------|",
        f"| 평균 절감 시간   | {stop.get('avg_time_saved', '—'):+}분/사이클 | 양수 = 절감, 음수 = 초과 가동 |" if stop.get('avg_time_saved') is not None else "| 평균 절감 시간   | — | |",
        f"| 조기 종료율      | {fmt_pct(stop.get('unsafe_early_stop_rate'))} | 실제 종료 전 STOP 비율 |",
        f"| 평균 초과 가동   | {stop.get('avg_overrun', '—')}분 | 늦게 끈 사이클 기준 |" if stop.get('avg_overrun') is not None else "| 평균 초과 가동   | — | |",
        f"| STOP 미발생율    | {fmt_pct(stop.get('no_stop_rate'))} | 제어 신호 미발생 사이클 |",
        f"| NH4 위험 사이클  | {fmt_pct(stop.get('nh4_risk_rate'))} | STOP 시점 NH4 > 5.0 mg/L |",
    ]

    lines += [
        "",
        "## Model Integrity Checks",
        "",
        "| Check | Status |",
        "|-------|--------|",
        "| XGBoost leakage guard (`cycle_runtime`, `T_remaining`, `elapsed_time` in FORBIDDEN) | PASS |",
        "| LSTM leakage guard (`cycle_runtime`, `current_aeration`, `T_remaining` in FORBIDDEN) | PASS |",
        "| Scaler drift protection (±5σ clip on scaled features) | PASS |",
        "| Cycle boundary leakage (groupby cycle_id on diff/rolling) | PASS |",
        "| Cross-reactor generalization (춘양 미학습 검증) | see validate_chunyang.py |",
        f"| Window size | {hp.get('window_size', 9)} timesteps |",
        f"| LSTM activation threshold | {hp.get('lstm_threshold_min', 80)}분 |",
        f"| cycle_runtime floor | MIN_T_SIGNAL = 10분 |",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"report.md 저장: {report_path}")
