FUTURE_HORIZON = 28
PAST_DAYS = 1

DATA_FILE_PATH = "data/panel.csv"
ML_MODEL_CHECKPOINT = "checkpoints/tft_best_train_final.pt"
PREDICTION_RESULTS_DIR = "checkpoints/"

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
