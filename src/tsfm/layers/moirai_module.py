#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from functools import partial

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from hydra.utils import instantiate
from jaxtyping import Bool, Float, Int
from torch import nn
from torch.distributions import Distribution
from torch.utils._pytree import tree_map

from uni2ts.module.ts_embed import MultiOutSizeLinear
from uni2ts.common.torch_util import mask_fill, packed_attention_mask
from uni2ts.distribution import DistributionOutput
from uni2ts.module.norm import RMSNorm
from uni2ts.module.packed_scaler import PackedNOPScaler, PackedScaler
from uni2ts.common.torch_util import safe_div
from uni2ts.module.position import (
    BinaryAttentionBias,
    QueryKeyProjection,
    RotaryProjection,
)
from uni2ts.module.transformer import TransformerEncoder
from uni2ts.module.ts_embed import MultiInSizeLinear
from einops import reduce


class PackedStdScaler(PackedScaler):
    def __init__(self, correction: int = 1, minimum_scale: float = 1e-5):
        super().__init__()
        self.correction = correction
        self.minimum_scale = minimum_scale

    def _get_loc_scale(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> tuple[
        Float[torch.Tensor, "*batch 1 #dim"], Float[torch.Tensor, "*batch 1 #dim"]
    ]:
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        tobs = reduce(
            id_mask * reduce(observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loc = reduce(
            id_mask * reduce(target * observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loc = safe_div(loc, tobs)
        var = reduce(
            id_mask
            * reduce(
                ((target - loc) ** 2) * observed_mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        var = safe_div(var, (tobs - self.correction))
        scale = torch.sqrt(var + self.minimum_scale)
        sample_id = sample_id.unsqueeze(-1)
        loc.masked_fill_(sample_id == 0, 0)
        scale.masked_fill_(sample_id == 0, 1)
        # loc[sample_id == 0] = 0
        # scale[sample_id == 0] = 1
        return loc, scale

def encode_distr_output(
    distr_output: DistributionOutput,
) -> dict[str, str | float | int]:
    """Serialization function for DistributionOutput"""

    def _encode(val):
        if not isinstance(val, DistributionOutput):
            return val

        return {
            "_target_": f"{val.__class__.__module__}.{val.__class__.__name__}",
            **tree_map(_encode, val.__dict__),
        }

    return _encode(distr_output)


def decode_distr_output(config: dict[str, str | float | int]) -> DistributionOutput:
    """Deserialization function for DistributionOutput"""
    return instantiate(config, _convert_="all")


def replace_linear(parent: nn.Module, module_name, weight, bias, is_input: bool, patch_size):
    module = nn.Linear(weight.shape[1], weight.shape[0],
                       bias=bias is not None, dtype=weight.dtype, device=weight.device)
    module.weight.data = weight
    if bias is not None:
        module.bias.data = bias
    module.register_forward_pre_hook(lambda module, args: (args[0][..., :patch_size] if is_input else args[0],))
    parent.register_module(module_name, module)


class MoiraiModule(
    nn.Module,
    PyTorchModelHubMixin,
    coders={DistributionOutput: (encode_distr_output, decode_distr_output)},
):
    """
    Contains components of Moirai, to ensure implementation is identical across models.
    Subclasses huggingface_hub.PyTorchModelHubMixin to support loading from HuggingFace Hub.
    """

    def __init__(
        self,
        distr_output: DistributionOutput,
        d_model: int,
        num_layers: int,
        patch_sizes: tuple[int, ...],  # tuple[int, ...] | list[int]
        max_seq_len: int,
        attn_dropout_p: float,
        dropout_p: float,
        scaling: bool = True,
    ):
        """
        :param distr_output: distribution output object
        :param d_model: model hidden dimensions
        :param num_layers: number of transformer layers
        :param patch_sizes: sequence of patch sizes
        :param max_seq_len: maximum sequence length for inputs
        :param attn_dropout_p: dropout probability for attention layers
        :param dropout_p: dropout probability for all other layers
        :param scaling: whether to apply scaling (standardization)
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_sizes = patch_sizes
        self.max_seq_len = max_seq_len
        self.scaling = scaling

        self.mask_encoding = nn.Embedding(num_embeddings=1, embedding_dim=d_model)
        self.scaler = PackedStdScaler() if scaling else PackedNOPScaler()
        self.in_proj = MultiInSizeLinear(
            in_features_ls=patch_sizes,
            out_features=d_model,
        )
        self.encoder = TransformerEncoder(
            d_model,
            num_layers,
            num_heads=None,
            pre_norm=True,
            attn_dropout_p=attn_dropout_p,
            dropout_p=dropout_p,
            norm_layer=RMSNorm,
            activation=F.silu,
            use_glu=True,
            use_qk_norm=True,
            var_attn_bias_layer=partial(BinaryAttentionBias),
            time_qk_proj_layer=partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection,
                kwargs=dict(max_len=max_seq_len),
                partial_factor=(0.0, 0.5),
            ),
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,
            d_ff=None,
        )
        self.distr_output = distr_output
        self.param_proj = self.distr_output.get_param_proj(d_model, patch_sizes)


    def focus_on_downstream(self, patch_size: int, start: int = None, end: int = None):
        # for layer in self.encoder.layers:
        #     layer.self_attn.var_attn_bias.register_forward_pre_hook(lambda module, args, kwargs:
        #                                                             ((arg[0] for arg in args),
        #                                                              {k: v[0] for k, v in kwargs.items()}),
        #                                                             with_kwargs=True)
        if start is not None or end is not None:
            self.param_proj.register_forward_pre_hook(lambda module, args: (args[0][..., start:end, :],
                                                                            *args[1:]))
            self.start, self.end = start, end

        if patch_size != 'auto':
            for i, in_feature in enumerate(self.in_proj.in_features_ls):
                if in_feature == patch_size:
                    replace_linear(self, 'in_proj', self.in_proj.weight[i][:, :patch_size],
                                   self.in_proj.bias[i] if self.in_proj.bias is not None else None, True, patch_size)
                    # self.in_proj.register_forward_pre_hook(lambda module, args: (args[0][..., :patch_size], args[1],))
                    # self.in_proj.in_features_ls = [in_feature]
                    # self.in_proj.weight.data = self.in_proj.weight[[i]][..., :patch_size]
                    # self.in_proj.mask = torch.tensor([True], device=self.in_proj.weight.device)
                    # if self.in_proj.bias is not None:
                    #     self.in_proj.bias.data = self.in_proj.bias[[i]]
                    break
            self.param_proj.out_size = patch_size
            for name, module in self.param_proj.named_modules():
                if isinstance(module, MultiOutSizeLinear):
                    for i, out_feature in enumerate(module.out_features_ls):
                        if out_feature // module.dim == patch_size:
                            parent = self.param_proj.get_submodule(".".join(name.split('.')[:-1]))
                            replace_linear(parent, name.split('.')[-1],
                                           module.weight[i].view(module.dim, -1, module.weight.shape[-1])[:, :patch_size].reshape(out_feature, -1),
                                           module.bias[i].view(module.dim, -1)[:, :patch_size].reshape(-1) if module.bias is not None else None,
                                           False, patch_size)
                            # module.out_features_ls = [out_feature]
                            # module.weight.data = (module.weight[[i]]
                            #                       .view(1, module.dim, -1, module.weight.shape[-1])[..., :patch_size, :]
                            #                       .reshape(1, out_feature, -1))
                            # module.mask = torch.tensor([True], device=self.in_proj.weight.device)
                            # if module.bias is not None:
                            #     module.bias.data = (module.bias[[i]]
                            #                         .view(1, module.dim, -1)[..., :patch_size]
                            #                         .reshape(1, out_feature))
                            break

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
    ) -> Distribution:
        """
        Defines the forward pass of MoiraiModule.
        This method expects processed inputs.

        1. Apply scaling to observations
        2. Project from observations to representations
        3. Replace prediction window with learnable mask
        4. Apply transformer layers
        5. Project from representations to distribution parameters
        6. Return distribution object

        :param target: input data
        :param observed_mask: binary mask for missing values, 1 if observed, 0 otherwise
        :param sample_id: indices indicating the sample index (for packing)
        :param time_id: indices indicating the time index
        :param variate_id: indices indicating the variate index
        :param prediction_mask: binary mask for prediction horizon, 1 if part of the horizon, 0 otherwise
        :param patch_size: patch size for each token
        :return: predictive distribution
        """
        loc, scale = self.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )
        scaled_target = (target - loc) / scale
        if hasattr(self, 'start') and hasattr(self, 'end'):
            loc = loc[..., self.start:self.end, :]
            scale = scale[..., self.start:self.end, :]
        reprs = self.in_proj(scaled_target, patch_size)
        masked_reprs = mask_fill(reprs, prediction_mask, self.mask_encoding.weight)
        reprs = self.encoder(
            masked_reprs,
            packed_attention_mask(sample_id),
            time_id=time_id,
            var_id=variate_id,
        )
        distr_param = self.param_proj(reprs, patch_size)
        distr = self.distr_output.distribution(distr_param, loc=loc, scale=scale)
        return distr
