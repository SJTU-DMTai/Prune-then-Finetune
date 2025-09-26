
import numpy as np
import pandas as pd
from gluonts.dataset.split import InputDataset
from gluonts.itertools import batcher
from gluonts.model import QuantileForecast
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tsfm_public.models.tinytimemixer.modeling_tinytimemixer import TinyTimeMixerLayer, TinyTimeMixerMLP, logger

from layers.prune_mask import MaskedLayer
import torch
from einops import rearrange

from models.base import PrunableModel, GluonTSPredictor
from utils.dataset import process_time_series
from utils.pruning import merge_weights
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.get_model import TTM_LOW_RESOLUTION_MODELS_MAX_CONTEXT, get_model
from data_provider.gluonts_data_wrapper import TTM_MAX_FORECAST_HORIZON, RESOLUTION_MAP, ForecastDataset, \
    get_freq_mapping, impute_series


class Model(nn.Module, PrunableModel, GluonTSPredictor):
    def __init__(
        self,
        config,
        **kwargs
    ):
        super().__init__()
        # Load pretrained model
        kwargs = {}
        if 'ett' in config.data_path.lower():
            kwargs.update(head_dropout = 0.7)
        self.freq = config.freq
        self.term = config.term
        if not config.gift_eval:
            self.model = get_model(
                "ibm-research/ttm-research-r2",
                context_length=config.seq_len,
                prediction_length=config.pred_len,
                freq_prefix_tuning=False,
                freq=getattr(config, 'freq', None),
                prefer_l1_loss=False,
                **kwargs,
            )
        else:
            self._get_gift_model("ibm-granite/granite-timeseries-ttm-r2", config.seq_len, config.pred_len, config.freq, config)
        freq_token = TimeSeriesPreprocessor().get_frequency_token(getattr(config, 'freq', None))
        self.register_buffer('freq_token', torch.tensor([freq_token]), False)
        self.prediction_length = config.pred_len
        self.enable_loss_fn = False
        self.scale = True
        self.force_short_context = False
        self.past_feat_dynamic_real_exist = False
        self.use_mask = False
        self.insample_forecast = True
        self.insample_errors = None
        self.quantile_keys = [
            "0.1",
            "0.2",
            "0.3",
            "0.4",
            "0.5",
            "0.6",
            "0.7",
            "0.8",
            "0.9",
            "mean",
        ]

    def linear_probing_(self):
        self.requires_grad_(False)
        self.model.head.requires_grad_(True)

    def _get_gift_model(self, model_path: str, context_length: int, prediction_length: int, freq: str, config, **kwargs):
        """Get suitable TTM model based on context and forecast lengths.

        Args:
            model_path (str): Model card link.
            context_length (int): Context length.
        """
        self.model = None

        prefer_l1_loss = False
        prefer_longer_context = True
        freq_prefix_tuning = False
        force_return = "zeropad"
        if self.term == "short" and (
            str(self.freq).startswith("W")
            or str(self.freq).startswith("M")
            or str(self.freq).startswith("Q")
            or str(self.freq).startswith("A")
        ):
            prefer_l1_loss = True
            prefer_longer_context = False
            freq_prefix_tuning = True

        if self.term == "short" and str(self.freq).startswith("D"):
            prefer_l1_loss = True
            freq_prefix_tuning = True
            if context_length < 2 * TTM_LOW_RESOLUTION_MODELS_MAX_CONTEXT:
                prefer_longer_context = False
            else:
                prefer_longer_context = True

        if self.term == "short" and str(self.freq).startswith("A"):
            self.insample_use_train = False
            self.use_valid_from_train = False
            force_return = "random_init_small"

        if prediction_length > TTM_MAX_FORECAST_HORIZON:
            force_return = "rolling"

        self.model = get_model(
            model_path=model_path,
            context_length=context_length,
            prediction_length=prediction_length,
            prefer_l1_loss=prefer_l1_loss,
            prefer_longer_context=prefer_longer_context,
            # resolution=RESOLUTION_MAP.get(freq, "oov"),
            freq_prefix_tuning=freq_prefix_tuning,
            force_return=force_return,
            **kwargs,
        )

        self.context_length = self.model.config.context_length
        config.seq_len = self.context_length
        print("Context length is updated to", config.seq_len)

        self.enable_prefix_tuning = False
        if hasattr(self.model.config, "resolution_prefix_tuning"):
            self.enable_prefix_tuning = self.model.config.resolution_prefix_tuning
        print(f"The TTM has Prefix Tuning = {self.enable_prefix_tuning}")
        config.enable_prefix_tuning = self.enable_prefix_tuning

    def forward(self, inputs, mask=None, labels=None):
        return self.model(inputs, past_observed_mask=mask,
                          freq_token=self.freq_token.expand(inputs.size(0), 1, 1, -1)).prediction_outputs

    def merge_weights_(self):
        dependency_graph = {
            'fc1': ({}, {'fc2': 0}),
            'fc2': ({'fc1': 1}, {}),
        }
        for module in self.model.modules():
            if isinstance(module, TinyTimeMixerMLP):
                merge_weights(module,
                              names=list(dependency_graph.keys()),
                              qkvo_names=[],
                              dependency_graph=dependency_graph,
                              num_heads=-1,
                              revise_head_num=None, enable_index_add=False,
                              prune_self_attention=None
                              )
        for module in self.model.modules():
            if isinstance(module, MaskedLayer) and not (hasattr(module, 'merged') and module.merged):
                merge_weights(module)


    def predict(
        self,
        test_data_input: InputDataset,
        batch_size: int = 64,
    ):
        """Predict.

        Args:
            test_data_input (InputDataset): Test input dataset.
            batch_size (int, optional): Batch size. Defaults to 64.

        Returns:
            float: Eval loss.
        """
        # We do not truncate the initial NaNs during testing since it sometimes
        # results in extremely short length, and inference fails.
        # Hence, in the current implementation the initial NaNs will be converted to zeros.
        test_data_input_scaled = process_time_series(test_data_input, truncate=False)
        # Standard scale
        if self.scale:
            test_data_input_scaled = self.scaler.transform(test_data_input_scaled)

        self.device = next(self.model.parameters()).device

        # Generate forecast samples
        forecast_samples = []
        series_ids = []
        for batch in tqdm(batcher(test_data_input_scaled, batch_size=batch_size)):
            batch_ttm = {}
            adjusted_batch_raw = []
            past_observed_mask = []
            for idx, entry in enumerate(batch):
                series_ids.append(entry["item_id"])

                # univariate array of shape (time,)
                # multivariate array of shape (var, time)
                # TTM supports multivariate time series
                if len(entry["target"].shape) == 1:
                    entry["target"] = entry["target"].reshape(1, -1)

                if self.force_short_context:
                    entry["target"] = entry["target"][:, -self.min_context_mult * self.prediction_length :]

                entry_context_length = entry["target"].shape[1]
                num_channels = entry["target"].shape[0]

                # Pad
                if entry_context_length < self.model.config.context_length:
                    logger.debug("Using zero filling for padding.")
                    # Zero-padding
                    padding = torch.zeros(
                        (
                            num_channels,
                            self.model.config.context_length - entry_context_length,
                        )
                    )
                    adjusted_entry = torch.cat(
                        (
                            padding,
                            torch.tensor(impute_series(entry["target"])),
                        ),
                        dim=1,
                    )
                    mask = torch.ones(adjusted_entry.shape)
                    mask[:, : padding.shape[1]] = 0

                    # observed_mask[idx, :, :(ttm.config.context_length - entry_context_length)] = 0
                # Truncate
                elif entry_context_length > self.model.config.context_length:
                    adjusted_entry = torch.tensor(
                        impute_series(entry["target"][:, -self.model.config.context_length :])
                    )
                    mask = torch.ones(adjusted_entry.shape)
                # Take full context
                else:
                    adjusted_entry = torch.tensor(impute_series(entry["target"]))
                    mask = torch.ones(adjusted_entry.shape)

                adjusted_batch_raw.append(adjusted_entry)
                past_observed_mask.append(mask.bool())

            # For TTM channel dimension comes at the end
            batch_ttm["past_values"] = torch.stack(adjusted_batch_raw).permute(0, 2, 1).to(self.device)
            if self.use_mask:
                batch_ttm["past_observed_mask"] = (
                    torch.stack(past_observed_mask).permute(0, 2, 1).to(self.device)
                )
            if self.model.config.resolution_prefix_tuning:
                freq_map = get_freq_mapping()
                batch_ttm["freq_token"] = (
                    torch.ones((batch_ttm["past_values"].shape[0])) * freq_map[self.freq]
                ).to("cuda")

            if self.prediction_length > TTM_MAX_FORECAST_HORIZON:
                batch_ttm["return_loss"] = False

                recursive_steps = int(np.ceil(self.prediction_length / self.model.config.prediction_length))
                predict_outputs = torch.empty(len(batch), 0, num_channels).to(self.device)
                with torch.no_grad():
                    for i in range(recursive_steps):
                        model_outputs = self.model(**batch_ttm)
                        batch_ttm["past_values"] = torch.cat(
                            [
                                batch_ttm["past_values"],
                                model_outputs["prediction_outputs"],
                            ],
                            dim=1,
                        )[:, -self.model.config.context_length :, :]
                        if self.use_mask:
                            batch_ttm["past_observed_mask"] = torch.cat(
                                [
                                    batch_ttm["past_observed_mask"],
                                    torch.ones(model_outputs["prediction_outputs"].shape)
                                    .bool()
                                    .to(self.device),
                                ],
                                dim=1,
                            )[:, -self.model.config.context_length :, :]
                        predict_outputs = torch.cat(
                            [
                                predict_outputs,
                                model_outputs["prediction_outputs"][:, : self.model.config.prediction_length, :],
                            ],
                            dim=1,
                        )
                predict_outputs = predict_outputs[:, : self.prediction_length, :]
            else:
                model_outputs = self.model(**batch_ttm)
                predict_outputs = model_outputs.prediction_outputs

            # Accumulate all forecasts
            forecast_samples.append(predict_outputs.detach().cpu().numpy())

        # list to np.ndarray
        forecast_samples = np.concatenate(forecast_samples)

        if self.scale:
            # inverse scale
            if self.past_feat_dynamic_real_exist:
                forecast_samples = self.scaler.inverse_transform(
                    forecast_samples,
                    series_ids,
                    self.prediction_channel_indices,
                )
            else:
                forecast_samples = self.scaler.inverse_transform(forecast_samples, series_ids)

        if self.prediction_length > TTM_MAX_FORECAST_HORIZON:
            forecast_samples = forecast_samples[:, :, : self.num_prediction_channels]

        if self.insample_forecast:
            point_forecasts = np.expand_dims(forecast_samples, 1)

            self.quantiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

            # Assuming forecasts, scale, and self.quantiles are defined
            b, seq_len, no_channels = forecast_samples.shape

            if self.insample_errors is None:
                dummy_errors_ = []
                unq_series_ids = list(np.unique(series_ids))
                for _ in unq_series_ids:
                    dummy_errors_.append(np.ones((seq_len, no_channels)))
                self.insample_errors = pd.DataFrame(
                    {"item_id": unq_series_ids, "errors": dummy_errors_}
                ).set_index("item_id")["errors"]
                logger.warning("`insample_errors` is `None`. Using a dummy error of `np.ones()`")

            # happens for H > 720
            if self.insample_errors.iloc[0].shape[0] < self.prediction_length:
                for i in range(len(self.insample_errors)):
                    self.insample_errors.iloc[i] = np.concatenate(
                        (
                            self.insample_errors.iloc[i],
                            self.insample_errors.iloc[i][
                                -(self.prediction_length - self.insample_errors.iloc[i].shape[0]) :,
                                :,
                            ],
                        )
                    )

            logger.info(f"Making quantile forecasts for quantiles {self.quantiles}")
            all_quantile_forecasts = []

            dataset = ForecastDataset(
                forecast_samples,
                series_ids,
                self.insample_errors,
                point_forecasts,
                self.quantiles,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda x: (
                    np.stack([i[0] for i in x]),  # forecast_samples
                    np.stack([i[1] for i in x]),  # insample_errors
                    np.stack([i[2] for i in x]),  # point_forecasts
                ),
            )

            all_quantile_forecasts = self.compute_quantile_forecasts(dataloader, self.quantiles)

        forecast_samples = np.array(all_quantile_forecasts)
        if forecast_samples.shape[-1] == 1:
            forecast_samples = np.squeeze(forecast_samples, axis=-1)

        # Convert forecast samples into gluonts SampleForecast objects
        #   Array of size (num_samples, prediction_length) (1D case) or
        #   (num_samples, prediction_length, target_dim) (multivariate case)
        sample_forecasts = []
        for item, ts in zip(forecast_samples, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            sample_forecasts.append(
                QuantileForecast(
                    forecast_arrays=item,
                    start_date=forecast_start_date,
                    forecast_keys=self.quantile_keys,
                    item_id=ts["item_id"],
                )
            )

        return sample_forecasts


    def compute_quantile_forecasts(self, loader, quantiles):
        all_quantile_forecasts = []

        for batch in tqdm(loader, desc="Processing Batches"):
            forecast_samples, insample_errors, point_forecasts = batch

            insample_errors[insample_errors == 0] = 1e-5  # To prevent division by zero

            # Expand scales for quantiles
            batch_size, seq_len, no_channels = forecast_samples.shape
            num_quantiles = len(quantiles)

            scales = np.expand_dims(insample_errors, axis=1)  # Shape: (batch_size, 1, H, C)
            scales = np.tile(scales, (1, num_quantiles, 1, 1))  # Shape: (batch_size, num_quantiles, H, C)

            # Expand quantiles
            quantiles_expanded = np.reshape(quantiles, (1, num_quantiles, 1, 1))  # Shape: (1, num_quantiles, 1, 1)
            quantiles_expanded = np.tile(
                quantiles_expanded, (batch_size, 1, seq_len, no_channels)
            )  # Shape: (batch_size, num_quantiles, H, C)

            # Expand forecasts
            forecasts_expanded = np.expand_dims(forecast_samples, axis=1)  # Shape: (batch_size, 1, H, C)
            forecasts_expanded = np.tile(
                forecasts_expanded, (1, num_quantiles, 1, 1)
            )  # Shape: (batch_size, num_quantiles, H, C)

            # Compute quantile forecasts
            from scipy.stats import norm
            quantile_forecasts = norm.ppf(
                quantiles_expanded, loc=forecasts_expanded, scale=scales
            )  # Shape: (batch_size, num_quantiles, H, C)

            # Append point forecasts
            final_forecasts = np.concatenate(
                (quantile_forecasts, point_forecasts), axis=1
            )  # Shape: (batch_size, num_quantiles+1, H, C)

            # Collect results for the batch
            all_quantile_forecasts.extend(final_forecasts)

        return all_quantile_forecasts