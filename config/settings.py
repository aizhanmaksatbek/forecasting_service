FUTURE_HORIZON = 28
PAST_DAYS = 1

DATA_FILE_PATH = "data/panel.csv"
ML_MODEL_DIR = "checkpoints/"
PRODUCT_NAME = "BEVERAGES"
STORE_NUMBER = 1


"""Configuration settings for TFT model."""
# WORKING_DIR = "/kaggle/input/demandForecasting"
WORKING_DIR = ""
RAW = f"{WORKING_DIR}/data/"

# encoder features
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
# known future features
DEC_VARS = [
    "onpromotion",
    "dow",
    "month",
    "weekofyear",
    "is_holiday",
    "is_workday"
]
# static features
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


import os


WORKING_DIR = ""
TFT_CHECKPOINTS_DIR = os.path.join(WORKING_DIR, "TFT", "checkpoints")
