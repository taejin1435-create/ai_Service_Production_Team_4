XGB_FEATURES = [
    "nh4",
    "no3",
    "ph",
    "temp",
    "상전류(R)",
    "nh4_no3_ratio",
    "do_saturation",
    "nh4_diff",
    "no3_diff",
    "ph_diff",
    "nh4_rolling_mean",
    "nh4_decay_rate",
    "hour_sin",
    "hour_cos",
    "weekday",
    "reactor",
]

XGB_MONOTONE_CONSTRAINTS = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)

XGB_TARGET = "cycle_runtime"

XGB_FORBIDDEN = {"elapsed_time", "T_remaining", "current_aeration", "cycle_runtime"}

LSTM_FEATURES = [
    "nh4",
    "no3",
    "ph",
    "temp",
    "상전류(R)",
    "nh4_no3_ratio",
    "do_saturation",
    "nh4_diff",
    "no3_diff",
    "ph_diff",
    "nh4_rolling_mean",
    "nh4_decay_rate",
    "hour_sin",
    "hour_cos",
    "weekday",
    "elapsed_time",
]

LSTM_TARGET = "T_remaining"

LSTM_FORBIDDEN = {"cycle_runtime", "current_aeration", "T_remaining"}
