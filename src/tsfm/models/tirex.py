import torch
from einops import rearrange
from torch import nn

from layers.prune_mask import MaskedLayer
from models.base import PrunableModel
from utils.pruning import merge_weights
from tirex import load_model


class Model(nn.Module, PrunableModel):
    def __init__(self, args):
        super(Model, self).__init__()
        self.model = load_model(f'NX-AI/TiRex', device='cpu')
        self.pred_len = args.pred_len
        self.batch_size = args.batch_size
        model_config = getattr(self.model, 'model_config', self.model.config)
        self.patch_len = self.input_patch_len = model_config.input_patch_size
        self.output_patch_len = model_config.output_patch_size
        self.n_covariate = args.n_covariate
        if args.n_covariate:
            self.covariate_emb = nn.Linear(args.n_covariate * self.patch_len * 2,
                                           self.model.out_norm.weight.shape[-1],
                                           bias=False)
            nn.init.zeros_(self.covariate_emb.weight)

    def forward(self, x: torch.FloatTensor = None,
                business_ids: torch.FloatTensor = None,
                weekday_ids: torch.FloatTensor = None,
                covariate_ids: torch.FloatTensor = None, labels: torch.FloatTensor = None):
        B, L, C = x.shape
        x = rearrange(x, "B L C -> (B C) L")
        if covariate_ids is not None:
            covariate_ids = rearrange(torch.nan_to_num(covariate_ids, 0), "B (L P) C -> B L (P C)", P=self.patch_len)
            covariate_ids = torch.cat((covariate_ids[:, 1:], covariate_ids[:, :-1]), dim=-1)
        mean = self._forecast_tensor(x, covariate_ids=covariate_ids, prediction_length=self.pred_len)[:, 4]
        mean = rearrange(mean, '(B C) L -> B L C', C=C)
        return mean

    def _forecast_tensor(
        self,
        context: torch.Tensor,
        covariate_ids: torch.LongTensor = None,
        prediction_length: int | None = None,
        max_context: int | None = None,
        max_accelerated_rollout_steps: int = 1,
    ) -> torch.Tensor:
        predictions = []
        remaining = -(self.pred_len // -self.model.tokenizer.patch_size)
        if max_context is None:
            max_context = self.model.config.train_ctx_len
        min_context = max(self.model.config.train_ctx_len, max_context)
        while remaining > 0:
            if context.shape[-1] > max_context:
                context = context[..., -max_context:]
            if context.shape[-1] < min_context:
                pad = torch.full(
                    (context.shape[0], min_context - context.shape[-1]),
                    fill_value=torch.nan,
                    device=context.device,
                    dtype=context.dtype,
                )
                context = torch.concat((pad, context), dim=1)
            tokenized_tensor, tokenizer_state = self.model.tokenizer.context_input_transform(context)
            fut_rollouts = min(remaining, max_accelerated_rollout_steps)
            prediction, _ = self._forward_model_tokenized(input_token=tokenized_tensor, rollouts=fut_rollouts,
                                                          covariate_ids=covariate_ids[:, :covariate_ids.shape[1]
                                                                                       - self.patch_len * (remaining - 1)]
                                                          if covariate_ids is not None else None)
            prediction = prediction[:, :, -fut_rollouts:, :].to(tokenized_tensor)  # predicted token
            # [bs, num_quantiles, num_predicted_token, output_patch_size]
            prediction = self.model.tokenizer.output_transform(prediction, tokenizer_state)
            prediction = prediction.flatten(start_dim=2)

            predictions.append(prediction)
            remaining -= fut_rollouts

            if remaining <= 0:
                break

            context = torch.cat([context, prediction[:, 4, :].detach()], dim=-1)

        return torch.cat(predictions, dim=-1)[..., :prediction_length].to(
            dtype=torch.float32,
        )

    def _forward_model_tokenized(
        self,
        input_token,
        input_mask=None,
        rollouts=1,
        covariate_ids: torch.LongTensor = None,
    ):
        input_mask = (
            input_mask.to(input_token.dtype)
            if input_mask is not None
            else torch.isnan(input_token).logical_not().to(input_token.dtype)
        )
        assert rollouts >= 1
        bs, numb_ctx_token, token_dim = input_token.shape
        if rollouts > 1:
            input_token = torch.cat(
                (
                    input_token,
                    torch.full(
                        (bs, rollouts - 1, token_dim),
                        fill_value=torch.nan,
                        device=input_token.device,
                        dtype=input_token.dtype,
                    ),
                ),
                dim=1,
            )
            input_mask = torch.cat(
                (
                    input_mask,
                    torch.full(
                        (bs, rollouts - 1, token_dim),
                        fill_value=False,
                        device=input_mask.device,
                        dtype=input_mask.dtype,
                    ),
                ),
                dim=1,
            )
        input_token = torch.nan_to_num(input_token, nan=self.model.config.nan_mask_value)
        hidden_states = self.model.input_patch_embedding(torch.cat((input_token, input_mask), dim=2))
        if self.n_covariate and covariate_ids is not None:
            covar_embeds = self.covariate_emb(covariate_ids)
            hidden_states = torch.cat([hidden_states[:, :-covar_embeds.shape[1]],
                                      hidden_states[:, -covar_embeds.shape[1]:] + covar_embeds], dim=1)

        for block in self.model.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.model.out_norm(hidden_states)

        quantile_preds = self.model.output_patch_embedding(hidden_states)
        quantile_preds = torch.unflatten(quantile_preds, -1, (len(self.model.config.quantiles), self.model.config.output_patch_size))
        quantile_preds = torch.transpose(quantile_preds, 1, 2)  # switch quantile and num_token_dimension
        # quantile_preds: [batch_size, num_quantiles, num_token, output_patch_size]

        return quantile_preds, hidden_states

    def merge_weights_(self):
        for module in self.model.modules():
            if (module not in self.transformers and isinstance(module, MaskedLayer)
                    and not (hasattr(module, 'merged') and module.merged)):
                merge_weights(module)
