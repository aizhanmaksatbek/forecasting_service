import os
import pandas as pd
import datetime as dt


DATA_RAW_DIR = "data/"
PAST_DAYS = 56
DATA_TEST = f"{DATA_RAW_DIR}test_56_days.csv"


def make_calendar(date_min, date_max, date_col="date"):
    cal = pd.DataFrame({date_col: pd.date_range(date_min, date_max, freq="D")})
    cal["dow"] = cal[date_col].dt.dayofweek.astype("int8")
    cal["month"] = cal[date_col].dt.month.astype("int8")
    cal["weekofyear"] = cal[date_col].dt.isocalendar().week.astype("int16")
    return cal


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

# Create the prediction dates
frc_strt_date = last_dec_strt
frc_end_date = last_dec_strt + dt.timedelta(days=28)

for (store_nbr, family, state, cluster, store_type), _ in test_ds.groupby(
    ["store_nbr", "family", "state", "cluster", "store_type"]
):
    frc_cal = make_calendar(frc_strt_date, frc_end_date)
    frc_cal = frc_cal.assign(
        store_nbr=store_nbr,
        family=family,
        state=state,
        cluster=cluster,
        store_type=store_type
        )
    frc_cal = frc_cal.assign(
        id=0,
        sales=0,
        onpromotion=0,
        transactions=0,
        dcoilwtico=0,
        is_holiday=0,
        is_workday=0
    )
    test_ds = pd.concat([test_ds, frc_cal], ignore_index=True)

test_ds.to_csv(DATA_TEST, index=False)
