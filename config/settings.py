FUTURE_HORIZON = 28
PAST_DAYS = 56

DATA_FILE_PATH = "database/test_56_days.csv"
DATA_RAW_DIR = "database/"
TFT_OUT_DIR = "database/"
ML_MODEL_CHECKPOINT = "checkpoints/tft_best_train_final.pt"
RESULTS_DIR = "checkpoints/"

ENC_VARS = [
    "sales",
    "transactions",
    "dcoilwtico",
    "onpromotion",
    "dow",
    "month",
    "weekofyear",
    "is_holiday",
    "is_workday"
]

DEC_VARS = [
    "onpromotion",
    "dow",
    "month",
    "weekofyear",
    "is_holiday",
    "is_workday"
]

STATIC_COLS = [
    "store_nbr",
    "family",
    "state",
    "cluster"
    ]

REALS_TO_SCALE = [
    "transactions",
    "dcoilwtico"
    ]
