import abc
import warnings
from enum import Enum
from typing import Union, List

import numpy as np
import torch
from gluonts.itertools import batcher
from gluonts.model import QuantileForecast, Forecast, SampleForecast
from torch import nn
from tqdm import tqdm


class ForecastType(Enum):
    SAMPLES = "samples"
    QUANTILES = "quantiles"


class PrunableModel(abc.ABC):
    @property
    def transformers(self) -> Union[nn.ModuleList, List[nn.Module]]:
        """
        Locate stacked transformers.
        """
        raise NotImplementedError

    def merge_weights_(self):
        raise NotImplementedError

    def forward(self, x: torch.FloatTensor = None, observed_mask: torch.BoolTensor = None, labels: torch.FloatTensor = None):
        raise NotImplementedError


class GluonTSPredictor:
    def predict(self, test_data_input, batch_size: int = 1024) -> List[Forecast]:
        if getattr(self, "scaler", None) is not None:
            test_data_input = list(self.scaler.transform(test_data_input))
        predict_kwargs = (
            {"num_samples": self.num_samples}
            if self.forecast_type == ForecastType.SAMPLES
            else {}
        )
        device = next(self.parameters()).device
        forecast_outputs = []
        series_ids = []
        for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
            series_ids += [entry["item_id"] for entry in batch]
            context = [torch.tensor(entry["target"], device=device) for entry in batch]
            forecast_outputs.append(
                self.infer(
                    context,
                    prediction_length=self.pred_len,
                    **predict_kwargs,
                ).detach().cpu().numpy()
            )
        forecast_outputs = np.concatenate(forecast_outputs)
        if getattr(self, "scaler", None) is not None:
            # inverse scale
            forecast_outputs = np.array(self.scaler.inverse_transform(forecast_outputs, series_ids))

        # Convert forecast samples into gluonts Forecast objects
        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            if getattr(self, "forecast_type", ForecastType.QUANTILES) == ForecastType.SAMPLES:
                forecasts.append(
                    SampleForecast(samples=item, start_date=forecast_start_date)
                )
            else:
                quantile_keys = list(map(str, self.quantiles))
                if 'mean' not in quantile_keys:
                    quantile_keys.append('mean')
                    item = np.concatenate([item, item[[self.quantiles.index(0.5)]]])
                forecasts.append(
                    QuantileForecast(
                        forecast_arrays=item,
                        forecast_keys=quantile_keys,
                        start_date=forecast_start_date,
                        item_id=ts["item_id"],
                    )
                )

        return forecasts