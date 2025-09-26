import types

from torch import nn
import torch
import timesfm
from timesfm.torch import util
from timesfm.torch.util import revin

class Model(nn.Module):
    def __init__(
        self,
        config,
        **kwargs
    ):
        super().__init__()
        # Load pretrained model
        self.model = timesfm.TimesFM_2p5_200M_torch()
        self.model.load_checkpoint()
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=config.seq_len,
                max_horizon=config.pred_len,
                normalize_inputs=False,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        self.forecast_config = self.model.forecast_config
        self.model = self.model.model
        self.horizon_len = config.pred_len
        self.patch_len = self.input_patch_len = 32
        self.output_patch_len = 128
        self.context_len = config.seq_len
        self.enable_loss_fn = getattr(config, "autoregressive", False)
        self.loss_fn = nn.MSELoss()
        self.model.decode = types.MethodType(decode, self.model)

    def forward(self, inputs,
                business_ids: torch.FloatTensor = None,
                weekday_ids: torch.FloatTensor = None,
                covariate_ids: torch.FloatTensor = None, labels: torch.FloatTensor = None):
        inputs = inputs.squeeze(-1)
        predictions = self.compiled_decode(
            self.horizon_len,
            inputs,
            torch.zeros_like(inputs).bool(),
        )[0].unsqueeze(-1)
        return predictions

    def compiled_decode(self, horizon, inputs, masks):
        forecast_config = fc = self.forecast_config
        if horizon > fc.max_horizon:
            raise ValueError(
                "Horizon must be less than the max horizon."
                f" {horizon} > {fc.max_horizon}."
            )

        batch_size = inputs.shape[0]
        if fc.infer_is_positive:
            is_positive = torch.all(inputs >= 0, dim=-1, keepdim=True)
        else:
            is_positive = None

        if fc.normalize_inputs:
            mu = torch.mean(inputs, dim=-1, keepdim=True)
            sigma = torch.std(inputs, dim=-1, keepdim=True)
            inputs = revin(inputs, mu, sigma, reverse=False)
        else:
            mu, sigma = None, None

        pf_outputs, quantile_spreads, ar_outputs = self.model.decode(
          forecast_config.max_horizon, inputs, masks
        )
        to_cat = [pf_outputs[:, -1, ...]]
        if ar_outputs is not None:
            to_cat.append(ar_outputs.reshape(batch_size, -1, self.model.q))
        full_forecast = torch.cat(to_cat, dim=1)

        flip_quantile_fn = lambda x: torch.cat(
          [x[..., :1], torch.flip(x[..., 1:], dims=(-1,))], dim=-1
        )

        if fc.force_flip_invariance:
            flipped_pf_outputs, flipped_quantile_spreads, flipped_ar_outputs = (
                self.model.decode(forecast_config.max_horizon, -inputs, masks)
            )
            flipped_quantile_spreads = flip_quantile_fn(flipped_quantile_spreads)
            flipped_pf_outputs = flip_quantile_fn(flipped_pf_outputs)
            to_cat = [flipped_pf_outputs[:, -1, ...]]
            if flipped_ar_outputs is not None:
              to_cat.append(
                  flipped_ar_outputs.reshape(batch_size, -1, self.model.q)
              )
            flipped_full_forecast = torch.cat(to_cat, dim=1)
            quantile_spreads = (quantile_spreads - flipped_quantile_spreads) / 2
            pf_outputs = (pf_outputs - flipped_pf_outputs) / 2
            full_forecast = (full_forecast - flipped_full_forecast) / 2

        if fc.use_continuous_quantile_head:
            for quantile_index in [1, 2, 3, 4, 6, 7, 8, 9]:
              full_forecast[:, :, quantile_index] = (
                  quantile_spreads[:, : fc.max_horizon, quantile_index]
                  - quantile_spreads[:, : fc.max_horizon, 5]
                  + full_forecast[:, : fc.max_horizon, 5]
              )
        full_forecast = full_forecast[:, :horizon, :]

        if fc.return_backcast:
            full_backcast = pf_outputs[:, :-1, : self.model.p, :].reshape(
                batch_size, -1, self.model.q
            )
            full_forecast = torch.cat([full_backcast, full_forecast], dim=1)

        if fc.fix_quantile_crossing:
            for i in [4, 3, 2, 1]:
              full_forecast[:, :, i] = torch.where(
                  full_forecast[:, :, i] < full_forecast[:, :, i + 1],
                  full_forecast[:, :, i],
                  full_forecast[:, :, i + 1],
              )
            for i in [6, 7, 8, 9]:
              full_forecast[:, :, i] = torch.where(
                  full_forecast[:, :, i] > full_forecast[:, :, i - 1],
                  full_forecast[:, :, i],
                  full_forecast[:, :, i - 1],
              )

        if fc.normalize_inputs:
            full_forecast = revin(full_forecast, mu, sigma, reverse=True)

        if is_positive is not None:
            full_forecast = torch.where(
                is_positive[..., None],
                torch.maximum(full_forecast, torch.zeros_like(full_forecast)),
                full_forecast,
            )

        return full_forecast[..., 5], full_forecast


def decode(self, horizon: int, inputs, masks):
    batch_size, context = inputs.shape[0], inputs.shape[1]
    num_decode_steps = (horizon - 1) // self.o
    num_input_patches = context // self.p
    decode_cache_size = num_input_patches + num_decode_steps * self.m

    # Prefill
    patched_inputs = torch.reshape(inputs, (batch_size, -1, self.p))
    patched_masks = torch.reshape(masks, (batch_size, -1, self.p))

    # running stats
    n = torch.zeros(batch_size, device=inputs.device)
    mu = torch.zeros(batch_size, device=inputs.device)
    sigma = torch.zeros(batch_size, device=inputs.device)
    patch_mu = []
    patch_sigma = []
    for i in range(num_input_patches):
        (n, mu, sigma), _ = util.update_running_stats(
            n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
        )
        patch_mu.append(mu)
        patch_sigma.append(sigma)
    last_n, last_mu, last_sigma = n, mu, sigma
    context_mu = torch.stack(patch_mu, dim=1)
    context_sigma = torch.stack(patch_sigma, dim=1)

    decode_caches = [
        util.DecodeCache(
            next_index=torch.zeros(
                batch_size, dtype=torch.int32, device=inputs.device
            ),
            num_masked=torch.zeros(
                batch_size, dtype=torch.int32, device=inputs.device
            ),
            key=torch.zeros(
                batch_size,
                decode_cache_size,
                self.h,
                self.hd,
                device=inputs.device,
            ),
            value=torch.zeros(
                batch_size,
                decode_cache_size,
                self.h,
                self.hd,
                device=inputs.device,
            ),
        )
        for _ in range(self.x)
    ]

    normed_inputs = revin(
        patched_inputs, context_mu, context_sigma, reverse=False
    )
    normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
    (_, _, normed_outputs, normed_quantile_spread), decode_caches = self(
        normed_inputs, patched_masks, decode_caches
    )
    renormed_outputs = torch.reshape(
        revin(normed_outputs, context_mu, context_sigma, reverse=True),
        (batch_size, -1, self.o, self.q),
    )
    renormed_quantile_spread = torch.reshape(
        revin(
            normed_quantile_spread, context_mu, context_sigma, reverse=True
        ),
        (batch_size, -1, self.os, self.q),
    )[:, -1, ...]

    # Autogressive decode
    ar_outputs = []
    last_renormed_output = renormed_outputs[:, -1, :, self.aridx]

    for _ in range(num_decode_steps):
        new_patched_input = torch.reshape(
            last_renormed_output, (batch_size, self.m, self.p)
        )
        new_mask = torch.zeros_like(new_patched_input, dtype=torch.bool)

        n, mu, sigma = last_n, last_mu, last_sigma
        new_mus, new_sigmas = [], []
        for i in range(self.m):
            (n, mu, sigma), _ = util.update_running_stats(
                n, mu, sigma, new_patched_input[:, i], new_mask[:, i]
            )
            new_mus.append(mu)
            new_sigmas.append(sigma)
        last_n, last_mu, last_sigma = n, mu, sigma
        new_mu = torch.stack(new_mus, dim=1)
        new_sigma = torch.stack(new_sigmas, dim=1)

        new_normed_input = revin(
            new_patched_input, new_mu, new_sigma, reverse=False
        )
        (_, _, new_normed_output, _), decode_caches = self(
            new_normed_input, new_mask, decode_caches
        )

        new_renormed_output = torch.reshape(
            revin(new_normed_output, new_mu, new_sigma, reverse=True),
            (batch_size, self.m, self.o, self.q),
        )
        ar_outputs.append(new_renormed_output[:, -1, ...])
        last_renormed_output = new_renormed_output[:, -1, :, self.aridx]

    if num_decode_steps > 0:
        ar_renormed_outputs = torch.stack(ar_outputs, dim=1)
    else:
        ar_renormed_outputs = None

    return renormed_outputs, renormed_quantile_spread, ar_renormed_outputs