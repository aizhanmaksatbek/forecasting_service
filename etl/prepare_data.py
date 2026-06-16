import os
import pandas as pd
import datetime as dt


DATA_RAW_DIR = "data/"
PAST_DAYS = 56
DATA_TEST = f"{DATA_RAW_DIR}test_56_days.csv"


def load_csv(name, parse_dates=None):
    path = os.path.join(DATA_RAW_DIR, name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=parse_dates)
    df["date"] = pd.to_datetime(df["date"])
    return df


df = load_csv("panel.csv")

end_date = df["date"].max()
last_dec_strt = end_date - dt.timedelta(days=PAST_DAYS)


test_ds = df[df["date"] >= last_dec_strt]

print(last_dec_strt)
print(end_date)
print(test_ds)

test_ds.to_csv(DATA_TEST, index=False)
