from typing import Union, List

import torch
from gluonts.torch.distributions import AffineTransformed
from torch import nn
import numpy as np
from typing import cast
from einops import repeat
from torch.distributions import Distribution

from layers.prune_mask import MaskedLayer
from models.base import PrunableModel
from utils.pruning import merge_weights
from toto.model.toto import Toto
from utils.toto_util import replace_extreme_values, pad_array, pad_id_mask, MaskedTimeseries

def loss(distr: torch.distributions.StudentT, label: torch.FloatTensor):
    return - 0.57 * distr.log_prob(label) + 0.43 * torch.log((distr.mean - label) ** 2 * 50 + 1)

class Model(nn.Module, PrunableModel):
    def __init__(self, args):
        super(Model, self).__init__()
        self.model = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0').model
        self.pred_len = args.pred_len
        self.batch_size = args.batch_size
        self.enable_loss_fn = False

    def forward(self, x: torch.FloatTensor = None,
                business_ids: torch.FloatTensor = None,
                weekday_ids: torch.FloatTensor = None,
                covariate_ids: torch.FloatTensor = None,
                labels: torch.FloatTensor = None, id_mask = None):
        x = x.transpose(-1, -2)
        dummy = torch.zeros_like(x)
        inputs = MaskedTimeseries(series=x, padding_mask=dummy.bool(),
                                  id_mask=dummy if id_mask is None else id_mask,
                                  timestamp_seconds=None, time_interval_seconds=None)
        predictions = self.forecast(inputs, prediction_length=self.pred_len, use_kv_cache=False).transpose(-1, -2)
        if labels is not None:
            return predictions, loss(predictions, labels)
        return predictions

    def forecast(
        self,
        inputs: MaskedTimeseries,
        prediction_length: int,
        use_kv_cache: bool = True,
    ):
        """
        Generate a forecast for a batch of time series. This method works autoregressively,
        i.e. it feeds the model's predictions back into itself. The decoding process is as follows:

        1. The model first processes the entire input context (historical data)
        2. For each future time step:
            - The model generates a distribution over possible values
            - Either the mean or random samples are drawn from this distribution
            - The generated value(s) are appended to the input sequence
            - The process repeats with this extended sequence

        Args:
            inputs: A MaskedTimeseries object containing the input time series.
            prediction_length: The number of future time steps to predict.
            num_samples:
                The number of samples to generate.
                If None, a single mean prediction is generated. However,
                the mean point forecast tends to be less accurate than the
                median or mean of the samples (provided enough samples are generated).
                It's recommended to use at least 128 samples for reliable forecasts.
            samples_per_batch:
                The number of samples to generate per batch.
                In most cases, this should be as high as possible, subject to memory constraints.
                When the inputs have a batch dimension, the effective batch size is samples_per_batch * batch_size.
            use_kv_cache:
                Whether to use a key-value cache for the model. In most cases, this should be True,
                as it significantly speeds up inference.
        """
        if len(inputs.series.shape) == 2:
            # unbatched input, variates x time_steps
            batch = cast(MaskedTimeseries, torch.utils.data.default_collate([inputs]))
        else:
            # input is already batched
            batch = inputs

        # pad the input to the nearest multiple of the patch size
        inputs = pad_array(batch.series, self.model.patch_embed.stride)
        input_padding_mask = pad_array(batch.padding_mask, self.model.patch_embed.stride)
        id_mask = batch.id_mask
        if id_mask is not None:
            id_mask = pad_id_mask(batch.id_mask, self.model.patch_embed.stride)
        # timestamp_seconds = pad_array(batch.timestamp_seconds, self.model.patch_embed.stride)
        # time_interval_seconds: Int[torch.Tensor, "batch variate series_len"] = torch.as_tensor(
        #     batch.time_interval_seconds, device=inputs.device, dtype=torch.int
        # )
        if input_padding_mask is None:
            input_padding_mask = torch.ones_like(inputs, dtype=torch.bool, device=inputs.device)
        if id_mask is None:
            id_mask = torch.zeros_like(inputs, dtype=torch.int, device=inputs.device)

        ## round up the prediction length to the nearest multiple of the patch size
        patch_size = self.model.patch_embed.stride
        rounded_steps = int(np.ceil(prediction_length / patch_size) * patch_size)
        start_index = inputs.shape[-1]
        end_index = start_index + prediction_length

        # TODO: maybe pass in future masks, rather than making assumptions here?
        dummy_padding = torch.ones(
            (input_padding_mask.shape[0], input_padding_mask.shape[1], patch_size),
            device=inputs.device,
            dtype=torch.bool,
        )
        dummy_id_mask = repeat(
            id_mask[:, :, -1:],
            "batch variates 1 -> batch variates patch_size",
            patch_size=patch_size,
        )
        if use_kv_cache:
            kv_cache = self.model.allocate_kv_cache(
                batch_size=inputs.shape[0],
                num_variates=inputs.shape[1],
                max_time_steps=inputs.shape[2] + rounded_steps,
                device=inputs.device,
                dtype=inputs.dtype,
            )
        else:
            kv_cache = None

        scaling_prefix_length = inputs.shape[-1]

        for _ in range(rounded_steps // patch_size):
            base_distr, loc, scale = self.model(
                inputs=inputs,
                input_padding_mask=input_padding_mask,
                id_mask=id_mask,
                kv_cache=kv_cache,
                scaling_prefix_length=scaling_prefix_length,
            )
            distr = self.create_affine_transformed(base_distr, loc, scale)

            # We remove extreme values that can occur early in training
            # and cause validation metrics to be NaN
            samples = replace_extreme_values(distr.mean[:, :, -patch_size:])

            inputs = torch.cat([inputs, samples], dim=-1)
            id_mask = torch.cat([id_mask, dummy_id_mask], dim=-1)
            input_padding_mask = torch.cat([input_padding_mask, dummy_padding], dim=-1)
        return inputs[:, :, start_index:end_index]

    @staticmethod
    def create_affine_transformed(base_distr: Distribution, loc: torch.Tensor, scale: torch.Tensor) -> Distribution:
        """
        Creates an AffineTransformed distribution with correctly matched shapes.

        Handles three cases:
        1. When loc/scale are per-timestep (from CausalStdMeanScaler)
        2. When base_distr only contains the distribution for the latest patch
           while loc/scale contain values for the entire sequence
        3. When loc/scale have a single time step (from StdMeanScaler/StdMinScaler)
           and need to be broadcast to match a multi-step base distribution

        Args:
            base_distr: The base distribution to transform
            loc: Location parameter
            scale: Scale parameter

        Returns:
            An AffineTransformed distribution with properly handled shapes
        """
        # Get the shape of the base distribution
        # We'll use this to match the time dimension of loc/scale
        base_shape = base_distr.mean.shape

        base_time_dim = base_shape[-1]  # Time dimension of base distribution
        loc_time_dim = loc.shape[-1]  # Time dimension of loc

        if loc_time_dim == 1:
            # Case 1: If loc/scale have time dimension 1 (standard scalers), PyTorch broadcasting will handle it
            return AffineTransformed(base_distr, loc=loc, scale=scale)

        # Case 2: If loc/scale have time dimension > 1 (causal scaler with history)
        # We need to extract only the suffix that matches the base distribution
        return AffineTransformed(base_distr, loc=loc[:, :, -base_time_dim:], scale=scale[:, :, -base_time_dim:])

    @property
    def transformers(self) -> Union[nn.ModuleList, List[nn.Module]]:
        return self.model.transformers

    def merge_weights_(self):
        for module in self.model.modules():
            if (module not in self.transformers and isinstance(module, MaskedLayer)
                    and not (hasattr(module, 'merged') and module.merged)):
                merge_weights(module)
