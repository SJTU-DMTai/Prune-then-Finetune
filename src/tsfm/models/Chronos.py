import warnings
from typing import Optional, Union, List

from chronos.utils import left_pad_and_stack_1D
from torch import nn

from layers.prune_mask import MaskedLayer
import torch

from models.base import PrunableModel, GluonTSPredictor
from utils.pruning import merge_weights, Zero
from chronos import ChronosBoltPipeline


class Model(nn.Module, PrunableModel, GluonTSPredictor):
    def __init__(
        self,
        config,
        **kwargs
    ):
        super().__init__()
        # Load pretrained model
        print("Load pretrained model from", f"amazon/chronos-bolt-{config.model_size}")
        self.model = ChronosBoltPipeline.from_pretrained(f"amazon/chronos-bolt-{config.model_size}",
                                                         torch_dtype=torch.bfloat16).model
        self.model.config.chronos_config["context_length"] = config.seq_len
        self.pred_len = config.pred_len
        self.mean_idx = self.model.config.chronos_config["quantiles"].index(0.5)
        self.enable_loss_fn = True
        self.transformer_names = ['q', 'k', 'v', 'o', None, 'wi', 'wo']

    @property
    def quantiles(self):
        return self.model.config.chronos_config["quantiles"]

    def linear_probing_(self):
        self.requires_grad_(False)
        self.model.output_patch_embedding.requires_grad_(True)

    def forward(self, inputs, observed_mask=None, labels=None):
        if observed_mask is not None:
            inputs = torch.masked_fill(inputs, ~observed_mask, torch.nan)
        inputs = inputs.squeeze(-1)
        if self.training:
            if labels is not None:
                labels.squeeze_(-1)
            outputs = self.model(inputs, target=labels)
            if labels is not None:
                return outputs.quantile_preds[..., self.mean_idx, :].unsqueeze(-1), outputs.loss
            else:
                predictions = outputs.quantile_preds
        else:
            predictions = self.infer(inputs, prediction_length=self.pred_len)
            predictions = predictions[..., self.mean_idx, :]
        return predictions.unsqueeze(-1)

    def infer(  # type: ignore[override]
            self,
            context: Union[torch.Tensor, List[torch.Tensor]],
            prediction_length: Optional[int] = None,
            limit_prediction_length: bool = False,
    ) -> torch.Tensor:
        """
        Get forecasts for the given time series.

        Refer to the base method (``BaseChronosPipeline.predict``)
        for details on shared parameters.
        Additional parameters
        ---------------------
        limit_prediction_length
            Force prediction length smaller or equal than the
            built-in prediction length from the model. False by
            default. When true, fail loudly if longer predictions
            are requested, otherwise longer predictions are allowed.

        Returns
        -------
        torch.Tensor
            Forecasts of shape (batch_size, num_quantiles, prediction_length)
            where num_quantiles is the number of quantiles the model has been
            trained to output. For official Chronos-Bolt models, the value of
            num_quantiles is 9 for [0.1, 0.2, ..., 0.9]-quantiles.

        Raises
        ------
        ValueError
            When limit_prediction_length is True and the prediction_length is
            greater than model's trainig prediction_length.
        """
        context_tensor = self._prepare_and_validate_context(context=context)

        model_context_length = self.model.config.chronos_config["context_length"]
        model_prediction_length = self.model.config.chronos_config["prediction_length"]
        if prediction_length is None:
            prediction_length = model_prediction_length

        if prediction_length > model_prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {model_prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)

        predictions = []
        remaining = prediction_length

        # We truncate the context here because otherwise batches with very long
        # context could take up large amounts of GPU memory unnecessarily.
        if context_tensor.shape[-1] > model_context_length:
            context_tensor = context_tensor[..., -model_context_length:]

        # TODO: We unroll the forecast of Chronos Bolt greedily with the full forecast
        # horizon that the model was trained with (i.e., 64). This results in variance collapsing
        # every 64 steps.
        context_tensor = context_tensor.to(
            device=self.model.device,
            dtype=torch.float32,
        )
        while remaining > 0:
            with torch.no_grad():
                prediction = self.model(
                    context=context_tensor,
                ).quantile_preds.to(context_tensor)

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            central_idx = torch.abs(torch.tensor(self.quantiles) - 0.5).argmin()
            central_prediction = prediction[:, central_idx]

            context_tensor = torch.cat([context_tensor, central_prediction], dim=-1)

        return torch.cat(predictions, dim=-1)[..., :prediction_length].to(dtype=torch.float32)

    def _prepare_and_validate_context(
        self, context: Union[torch.Tensor, List[torch.Tensor]]
    ):
        if isinstance(context, list):
            context = left_pad_and_stack_1D(context)
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2

        return context

    @property
    def transformers(self):
        return ([block for block in self.model.encoder.block] +
                [block for block in self.model.decoder.block])

    def merge_weights_(self):
        encoder_dependency_graph = {
            'layer.0.SelfAttention.q': ({}, {'layer.0.SelfAttention.k': 1}),
            'layer.0.SelfAttention.k': ({}, {'layer.0.SelfAttention.q': 1}),
            'layer.0.SelfAttention.v': ({}, {'layer.0.SelfAttention.o': 0}),
            'layer.0.SelfAttention.o': ({'layer.0.SelfAttention.v': 1}, {}),
            'layer.1.DenseReluDense.wi': ({}, {'layer.1.DenseReluDense.wo': 0}),
            'layer.1.DenseReluDense.wo': ({'layer.1.DenseReluDense.wi': 1}, {}),
        } # 0 is input dimension, 1 is output dimension
        decoder_dependency_graph = {
            'layer.0.SelfAttention.v': ({}, {'layer.0.SelfAttention.o': 0}),
            'layer.0.SelfAttention.o': ({'layer.0.SelfAttention.v': 1}, {}),
            'layer.1.EncDecAttention.q': ({}, {'layer.1.EncDecAttention.k': 1}),
            'layer.1.EncDecAttention.k': ({}, {'layer.1.EncDecAttention.q': 1}),
            'layer.1.EncDecAttention.v': ({}, {'layer.1.EncDecAttention.o': 0}),
            'layer.1.EncDecAttention.o': ({'layer.1.EncDecAttention.v': 1}, {}),
            'layer.2.DenseReluDense.wi': ({}, {'layer.2.DenseReluDense.wo': 0}),
            'layer.2.DenseReluDense.wo': ({'layer.2.DenseReluDense.wi': 1}, {}),
        } # 0 is input dimension, 1 is output dimension
        qkvo_names = ['layer.0.SelfAttention.q', 'layer.0.SelfAttention.k',
                      'layer.0.SelfAttention.v', 'layer.0.SelfAttention.o']
        decoder_qkvo_names = ['layer.1.EncDecAttention.q', 'layer.1.EncDecAttention.k',
                              'layer.1.EncDecAttention.v', 'layer.1.EncDecAttention.o']

        def revise_head_num(layer, discard_head):
            discard_head = discard_head.flatten()
            reduce_head_num = discard_head.sum()
            layer.layer[0].SelfAttention.n_heads -= reduce_head_num
            layer.layer[0].SelfAttention.inner_dim = layer.layer[0].SelfAttention.key_value_proj_dim * layer.layer[0].SelfAttention.n_heads
            layer.layer[0].SelfAttention.pruned_heads = torch.arange(len(discard_head), device=discard_head.device)[discard_head].tolist()

        def revise_decoder_head_num(layer, discard_head):
            # revise_head_num(layer, discard_head)
            discard_head = discard_head.flatten()
            reduce_head_num = discard_head.sum()
            layer.layer[1].EncDecAttention.n_heads -= reduce_head_num
            layer.layer[1].EncDecAttention.inner_dim = layer.layer[1].EncDecAttention.key_value_proj_dim * layer.layer[1].EncDecAttention.n_heads
            if reduce_head_num:
                layer.layer[1].EncDecAttention.pruned_heads = torch.arange(len(discard_head), device=discard_head.device)[discard_head].tolist()

        enable_index_add = False
        merge_weights(
            self.model.encoder.block,
            names=qkvo_names + ['layer.1.DenseReluDense.wi', 'layer.1.DenseReluDense.wo'],
            qkvo_names=qkvo_names,
            dependency_graph=encoder_dependency_graph,
            num_heads=self.model.encoder.block[0].layer[0].SelfAttention.n_heads,
            revise_head_num=revise_head_num,
            enable_index_add=enable_index_add, reduce_V=False, num_attn_outputs=3,
        )
        merge_weights(
            self.model.decoder.block,
            names=decoder_qkvo_names + ['layer.2.DenseReluDense.wi', 'layer.2.DenseReluDense.wo'],
            qkvo_names=decoder_qkvo_names,
            dependency_graph=decoder_dependency_graph,
            num_heads=self.model.decoder.block[0].layer[0].SelfAttention.n_heads,
            revise_head_num=revise_decoder_head_num,
            enable_index_add=enable_index_add, reduce_V=False, num_attn_outputs=3,
        )
        for block in self.model.decoder.block:
            block.layer[0].SelfAttention = Zero(block.layer[0].SelfAttention.d_model, 3)

        for module in self.model.modules():
            if (module not in self.transformers and isinstance(module, MaskedLayer)
                    and not (hasattr(module, 'merged') and module.merged)):
                merge_weights(module)
