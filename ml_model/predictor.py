import pandas as pd
import os
import numpy as np
import datetime as dt
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from ml_model.tft import TemporalFusionTransformer
from ml_model.tft_dataset import TFTWindowDataset, tft_collate
from ml_model.utils import build_onehot_maps
from config.settings import (
    ENC_VARS,
    DEC_VARS,
    STATIC_COLS,
    REALS_TO_SCALE,
    RESULTS_DIR,
    ML_MODEL_CHECKPOINT
    )


class Predictor():
    def __init__(self, input_data: pd.DataFrame):
        self._data = input_data
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.out_csv = os.path.join(RESULTS_DIR, f"rs_{dt.datetime.now()}.csv")

        assert os.path.exists(ML_MODEL_CHECKPOINT), (
            f"Checkpoint not found: {ML_MODEL_CHECKPOINT}"
            )
        ckpt = torch.load(ML_MODEL_CHECKPOINT, map_location=self.device)

        self.cfg = ckpt.get("cfg", {})
        self.quantiles = [float(quantile)
                          for quantile in self.cfg["quantiles"].split(",")
                          ]
        self.enc_len = self.cfg["enc_len"]
        self.dec_len = self.cfg["dec_len"]
        self.batch_size = self.cfg["batch_size"]
        self.stride = self.cfg["stride"]

        self.data_loader, self.static_dims = self.wrap_data_into_loader()

        # Build model to match checkpoint shapes
        self.model = TemporalFusionTransformer(
            static_input_dims=self.static_dims,
            past_input_dims=[1] * len(ENC_VARS),
            future_input_dims=[1] * len(DEC_VARS),
            d_model=self.cfg["d_model"],
            hidden_dim=self.cfg["hidden_dim"],
            n_heads=self.cfg["heads"],
            lstm_hidden_size=self.cfg["lstm_hidden"],
            lstm_layers=self.cfg["lstm_layers"],
            dropout=self.cfg["dropout"],
            num_quantiles=len(self.quantiles),
        ).to(self.device)

        # Load weights strictly
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        print(f"Loaded stored TFT model for evaluation {ML_MODEL_CHECKPOINT}")

    def predict(self):
        median_idx = int(np.argmin([abs(q - 0.5) for q in self.quantiles]))

        self.model.eval()
        rows = []
        test_preds = []

        with torch.no_grad():
            for batch in self.data_loader:
                past = batch["past_inputs"].to(self.device)
                future = batch["future_inputs"].to(self.device)
                static = batch["static_inputs"].to(self.device)

                out = self.model(past, future, static)
                preds_med = out["prediction"][..., median_idx]  # [B, L_dec]
                preds = preds_med.cpu().numpy()
                yhat = out["prediction"][..., median_idx]
                test_preds.append(yhat.detach().cpu().numpy())

                metas = batch.get("meta", [])
                for i, meta in enumerate(metas):
                    store_nbr = meta["store_nbr"]
                    family = meta["family"]
                    fut_dates = meta["future_dates"]
                    for d_idx, date in enumerate(fut_dates):
                        rows.append({
                            "date": pd.to_datetime(date),
                            "store_nbr": store_nbr,
                            "family": family,
                            "y_pred": float(preds[i, d_idx]),
                        })
        self.save_results_csv(rows)

    def save_results_csv(self, rows):
        forecasts_df = (
            pd.DataFrame(rows)
            .sort_values(["family", "store_nbr", "date"])
        )
        forecasts_df.to_csv(self.out_csv, index=False)
        print(f"Saved test forecasts CSV -> {self.out_csv}")

    def wrap_data_into_loader(self):
        """
        This function prepares Dataloader.
        """
        scaler = StandardScaler()
        self._data.loc[:, REALS_TO_SCALE] = scaler.fit_transform(
            self._data.loc[:, REALS_TO_SCALE]
        )

        static_maps = build_onehot_maps(self._data, STATIC_COLS)
        static_dims = [len(static_maps[c]) for c in STATIC_COLS]

        _ds = TFTWindowDataset(
            self._data, self.enc_len, self.dec_len,
            ENC_VARS, DEC_VARS, STATIC_COLS,
            stride=self.stride, static_onehot_maps=static_maps,
        )

        _ds_loader = DataLoader(
            _ds, batch_size=self.batch_size, shuffle=False,
            num_workers=4, collate_fn=tft_collate,
        )
        return (_ds_loader, static_dims)
