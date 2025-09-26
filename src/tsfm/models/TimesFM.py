from torch import nn

from layers.prune_mask import MaskedLayer
import torch

from models.base import PrunableModel
from utils.pruning import merge_weights
import timesfm
from timesfm import timesfm_base

class Model(nn.Module, PrunableModel):
    def __init__(
        self,
        config,
        **kwargs
    ):
        super().__init__()
        # Load pretrained model
        print("Load pretrained model from", "google/timesfm-2.0-500m-pytorch")
        self.model = timesfm.TimesFm(
              hparams=timesfm.TimesFmHparams(
                  backend='gpu',
                  per_core_batch_size=32,
                  horizon_len=config.pred_len,
                  input_patch_len=32,
                  output_patch_len=128,
                  num_layers=50,
                  model_dims=1280,
                  use_positional_embedding=False,
              ),
              checkpoint=timesfm.TimesFmCheckpoint(
                  huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
        )._model
        self.horizon_len = config.pred_len
        self.input_patch_len = 32
        self.output_patch_len = 128
        self.context_len = config.seq_len
        self.enable_loss_fn = getattr(config, "autoregressive", False)
        self.register_buffer('freq', torch.tensor([timesfm_base.freq_map(config.freq)]).int(), False)
        self.register_buffer('padding', torch.zeros((config.seq_len + config.pred_len, )), False)
        self.loss_fn = nn.MSELoss()


    def forward(self, inputs, labels=None):
        inputs.squeeze_(-1)
        freq = self.freq.expand(inputs.shape[0], -1)
        if self.training:
            paddings = self.padding[:self.context_len].expand(inputs.shape[0], -1)
            predictions = self.model(inputs, input_padding=paddings, freq=freq)[..., 0]
            if labels is not None:
                labels = labels.squeeze(-1).unfold(-1, self.output_patch_len, self.input_patch_len)
                loss = self.loss_fn(predictions[:, -labels.size(1):], labels)
                return predictions.unsqueeze(-1), loss
            else:
                return predictions[:, -1].unsqueeze(-1)
        else:
            paddings = self.padding.expand(inputs.shape[0], -1)
            predictions = self.model.decode(
                input_ts=inputs,
                paddings=paddings,
                freq=freq,
                horizon_len=self.horizon_len,
                output_patch_len=self.output_patch_len,
                return_forecast_on_context=False,
            )[0].unsqueeze(-1)
        return predictions

    @property
    def transformers(self):
        return self.model.stacked_transformer.layers

    def merge_weights_(self):
        dependency_graph = {
            'mlp.gate_proj': ({}, {'mlp.down_proj': 0}),
            'mlp.down_proj': ({'mlp.gate_proj': 1}, {}),
        } # 0 is input dimension, 1 is output dimension
        merge_weights(self.transformers,
                      names=['mlp.gate_proj', 'mlp.down_proj'],
                      qkvo_names=[],
                      dependency_graph=dependency_graph,
                      num_heads=self.transformers[0].self_attn.num_heads,
                      revise_head_num=None, enable_index_add=False,
                      prune_self_attention=None
                      )

        for module in self.model.modules():
            if (module not in self.transformers and isinstance(module, MaskedLayer)
                    and not (hasattr(module, 'merged') and module.merged)):
                merge_weights(module)
