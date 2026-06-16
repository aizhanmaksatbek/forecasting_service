import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from ml_model.TFT.architecture.tft import (
    TemporalFusionTransformer,
    QuantileLoss
    )
from ml_model.TFT.tft_dataset import TFTWindowDataset, tft_collate
from ml_model.TFT.utils import build_onehot_maps
from config.settings import (
    ENC_VARS,
    DEC_VARS,
    STATIC_COLS,
    REALS_TO_SCALE,
    PREDICTION_RESULTS_DIR,
    ML_MODEL_CHECKPOINT
    )


def save_results_csv(rows):
    if rows:
        test_forecasts_df = (
            pd.DataFrame(rows)
            .sort_values(["family", "store_nbr", "date"])
        )
        out_csv = os.path.join(PREDICTION_RESULTS_DIR, "forecast_results.csv")
        test_forecasts_df.to_csv(out_csv, index=False)
        print(f"Saved test forecasts CSV -> {out_csv}")


def wrap_data_into_loader(df, dec_len, enc_len, batch_size, stride):
    """
    This function prepares Dataloader.
    """
    scaler = StandardScaler()
    df.loc[:, REALS_TO_SCALE] = scaler.fit_transform(
        df.loc[:, REALS_TO_SCALE]
    )

    static_maps = build_onehot_maps(df, STATIC_COLS)
    static_dims = [len(static_maps[c]) for c in STATIC_COLS]

    _ds = TFTWindowDataset(
        df, enc_len, dec_len, ENC_VARS, DEC_VARS, STATIC_COLS,
        stride=stride, static_onehot_maps=static_maps,
    )

    _ds_loader = DataLoader(
        _ds, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=tft_collate,
    )
    return (_ds_loader, static_dims, len(_ds))


def eval_loader(model, data_loader, quantiles, test_len):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))

    # Important: do NOT load a checkpoint here; model is already loaded
    model.eval()
    criterion = QuantileLoss(quantiles=quantiles)
    rows = []
    total_loss = 0.0
    test_ys, test_preds = [], []

    with torch.no_grad():
        for batch in data_loader:
            past = batch["past_inputs"].to(device)
            future = batch["future_inputs"].to(device)
            static = batch["static_inputs"].to(device)
            y = batch["target"].to(device)

            out = model(past, future, static)
            preds_med = out["prediction"][..., median_idx]  # [B, L_dec]
            preds = preds_med.cpu().numpy()
            loss = criterion(out["prediction"].to(device), y)
            total_loss += loss.item() * past.size(0)
            yhat = out["prediction"][..., median_idx]
            test_ys.append(y.detach().cpu().numpy())
            test_preds.append(yhat.detach().cpu().numpy())

            metas = batch.get("meta", [])
            for i, meta in enumerate(metas):
                store_nbr = meta["store_nbr"]
                family = meta["family"]
                fut_dates = meta["future_dates"]
                targets = batch["target"].cpu().numpy()   # [B, L_dec]
                for d_idx, date in enumerate(fut_dates):
                    rows.append({
                        "date": pd.to_datetime(date),
                        "store_nbr": store_nbr,
                        "family": family,
                        "y_true": float(targets[i, d_idx]),
                        "y_pred": float(preds[i, d_idx]),
                    })
                sales_idx = ENC_VARS.index("sales")
                past_dates = meta["past_dates"]
                for d_idx, date in enumerate(past_dates):
                    rows.append({
                        "date": pd.to_datetime(date),
                        "store_nbr": store_nbr,
                        "family": family,
                        "y_past": float(past[i, d_idx, sales_idx].cpu()),
                    })
    print(rows)
    save_results_csv(rows)
    total_loss /= max(test_len, 1)
    print(f"Test loss: {total_loss:.4f}")


def make_forecast(input_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert os.path.exists(ML_MODEL_CHECKPOINT), (
        f"Checkpoint not found: {ML_MODEL_CHECKPOINT}"
        )
    ckpt = torch.load(ML_MODEL_CHECKPOINT, map_location=device)

    cfg = ckpt.get("cfg", {})
    quantiles = [float(quantile) for quantile in cfg["quantiles"].split(",")]
    enc_len = cfg["enc_len"]
    dec_len = cfg["dec_len"]
    batch_size = cfg["batch_size"]
    stride = cfg["stride"]

    test_loader, static_dims, test_len = wrap_data_into_loader(
        input_data,
        dec_len, enc_len, batch_size, stride
    )

    # Build model to match checkpoint shapes
    model = TemporalFusionTransformer(
        static_input_dims=static_dims,
        past_input_dims=[1] * len(ENC_VARS),
        future_input_dims=[1] * len(DEC_VARS),
        d_model=cfg["d_model"],
        hidden_dim=cfg["hidden_dim"],
        n_heads=cfg["heads"],
        lstm_hidden_size=cfg["lstm_hidden"],
        lstm_layers=cfg["lstm_layers"],
        dropout=cfg["dropout"],
        num_quantiles=len(quantiles),
    ).to(device)

    # Load weights strictly
    model.load_state_dict(ckpt["model_state"], strict=True)
    print(f"Loaded stored TFT model for evaluation {ML_MODEL_CHECKPOINT}")

    # Evaluate
    eval_loader(model, test_loader, quantiles, test_len)
