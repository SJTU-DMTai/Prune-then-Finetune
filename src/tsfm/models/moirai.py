import sys

from torch import nn

from layers.prune_mask import MaskedLayer
from models.base import PrunableModel
from uni2ts.loss.packed import PackedNLLLoss
from layers.moirai_module import MoiraiModule
from layers.moirai_forecast import MoiraiForecast
import torch
from einops import rearrange

from utils.pruning import merge_weights, prune_head_dim
from utils.tools import as_buffer_


class Model(nn.Module, PrunableModel):
    def __init__(
        self,
        config,
        **kwargs
    ):
        super().__init__()
        self.context_length = config.seq_len
        self.prediction_length = config.pred_len
        self.target_dim = config.target_dim if getattr(config, "mode", "M") == 'M' else 1
        # self.freq = config.freq
        self.num_samples = 100
        self.patch_size = int(config.patch_len)
        self.multivariate = getattr(config, 'mode', 'M') == 'M'

        print("Load pretrained model from", f"Salesforce/moirai-1.0-R-{config.model_size}")
        self.moirai = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{config.model_size}"),
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            patch_size=self.patch_size,
            target_dim=self.target_dim,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0
        )
        start = self.target_dim * self.moirai.context_token_length(self.patch_size)
        end = start + self.target_dim * self.moirai.prediction_token_length(self.patch_size)
        self.moirai.module.focus_on_downstream(self.patch_size, start, end)
        self.register_buffer("past_is_pad", torch.zeros(1, self.moirai.past_length).bool())
        self.register_buffer("past_observed_target", torch.ones(self.moirai.past_length, self.target_dim).bool())
        self.loss_func = PackedNLLLoss()

        # To avoid setting find_unused_parameters
        as_buffer_(self.moirai.module.param_proj.proj.components[0].scale, 'weight')
        as_buffer_(self.moirai.module.param_proj.proj.components[0].scale, 'bias')
        as_buffer_(self.moirai.module.param_proj.proj.components[0].df, 'weight')
        as_buffer_(self.moirai.module.param_proj.proj.components[0].df, 'bias')

        self.transformer_names =  ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj',
                                   'ffn.fc_gate', 'ffn.fc1', 'ffn.fc2']

    @property
    def transformers(self):
        return self.moirai.module.encoder.layers

    def forward(self, inputs, labels=None):
        B, _, K = inputs.shape
        if not self.multivariate:
            inputs = rearrange(inputs, 'b l k -> (b k) l 1') #(210,96,1)
        if self.training:
            forecast = self.moirai._get_distr(self.patch_size,
                                               past_target=inputs,
                                               past_observed_target=self.past_observed_target.expand(len(inputs), -1, -1),
                                               past_is_pad=self.past_is_pad,).mean
            if self.multivariate:
                forecast = rearrange(forecast, 'b (k l) p -> b (l p) k', b=B, k=K)[:, :self.prediction_length]
            else:
                forecast = rearrange(forecast, 'b l p -> b (l p) 1')[:, :self.prediction_length]
        else:
            with torch.no_grad():
                forecast = self.moirai._get_distr(self.patch_size,
                                                   past_target=inputs,
                                                   past_observed_target=self.past_observed_target.expand(len(inputs), -1, -1),
                                                   past_is_pad=self.past_is_pad,).mean
                if self.multivariate:
                    forecast = rearrange(forecast, 'b (k l) p -> b (l p) k', b=B, k=K)[:, :self.prediction_length]
                else:
                    forecast = rearrange(forecast, 'b l p -> b (l p) 1')[:, :self.prediction_length]
        return forecast

    def merge_weights_(self):
        dependency_graph = {
            # 'self_attn.k_proj': ({}, {'self_attn.q_proj': 0}), # cannot prune given q_norm
            # 'self_attn.q_proj': ({}, {'self_attn.k_proj': 0}), # cannot prune given k_norm
            'self_attn.v_proj': ({}, {'self_attn.out_proj': 0}),
            'self_attn.out_proj': ({'self_attn.v_proj': 1}, {}),
            'ffn.fc1': ({}, {'ffn.fc_gate': 1, 'ffn.fc2': 0}),
            'ffn.fc_gate': ({}, {'ffn.fc1': 1, 'ffn.fc2': 0}),
            'ffn.fc2': ({'ffn.fc_gate': 1, 'ffn.fc1': 1}, {}),
        } # 0 is input dimension, 1 is output dimension
        qkvo_names = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj']

        def revise_head_num(layer, discard_head):
            reduce_head_num = discard_head.sum()
            layer.self_attn.num_heads -= reduce_head_num
            layer.self_attn.num_groups -= reduce_head_num
            layer.self_attn.var_attn_bias.num_groups -= reduce_head_num
            layer.self_attn.var_attn_bias.num_heads -= reduce_head_num

            if layer.self_attn.var_attn_bias is not None:
                layer.self_attn.var_attn_bias.emb.weight.data = layer.self_attn.var_attn_bias.emb.weight[:,
                                                                ~discard_head[:, 0]]

        for k, module in sys.modules.items():
            if 'uni2ts.module.norm' in k:
                module.__dict__['RMSNorm'].forward = rms_norm_forward
        enable_index_add = False
        # try:
        #     for k, module in sys.modules.items():
        #         if 'uni2ts.module.transformer' in k:
        #             module.__dict__['TransformerEncoderLayer'].forward = forward
        #             enable_index_add = True
        # except Exception as e:
        #     print(e)
        #     enable_index_add = False

        for layer in self.transformers:
            if layer.self_attn.time_qk_proj is not None and layer.self_attn.time_qk_proj.partial_factor is not None:
                def register_layer_id(layer, args):
                    if hasattr(layer.self_attn, 'time_qk_proj'):
                        layer.self_attn.time_qk_proj.layer_id = layer.layer_id
                layer.register_forward_pre_hook(register_layer_id)

        merge_weights(
            self.transformers,
            names=self.transformer_names,
            qkvo_names=qkvo_names,
            dependency_graph=dependency_graph,
            num_heads=self.transformers[0].self_attn.num_heads,
            revise_head_num=revise_head_num,
            enable_index_add=enable_index_add,
            # prune_self_attention=partial(prune_self_attention,
            #                              num_heads=self.transformers[0].self_attn.num_heads,
            #                              qk_proj=self.transformers[0].self_attn.time_qk_proj,
            #                              qk_norm_names=['self_attn.q_norm', 'self_attn.k_norm'],)
        )

        for module in self.moirai.modules():
            if (module not in self.transformers and isinstance(module, MaskedLayer)
                    and not (hasattr(module, 'merged') and module.merged)):
                merge_weights(module)

def prune_self_attention(layer, discard_out, qkvo_names, num_heads, head_dim, qk_proj, qk_norm_names):
    q_zero_out = layer.get_submodule(qkvo_names[0]).mask_out.mask == 0
    k_zero_out = layer.get_submodule(qkvo_names[1]).mask_out.mask == 0

    qk_zero_out = torch.stack([q_zero_out, k_zero_out], dim=0).view(2, -1, qk_proj.head_dim)
    qk_zero_out = list(qk_zero_out.split(qk_proj.split_sizes, dim=-1))
    qk_zero_out[1][..., 0::2] = qk_zero_out[1][..., 1::2] = qk_zero_out[1][..., 0::2] & qk_zero_out[1][..., 1::2]
    qk_zero_out = [(zero_out[0] | zero_out[1]) for zero_out in qk_zero_out]

    qk_zero_out[1] = qk_zero_out[1].view(num_heads, -1, 2)
    discard_qk = [None, None]
    discard_qk[1] = prune_head_dim(qk_zero_out[1][..., 0].flatten(), num_heads, layer.layer_id, 'QK')
    if discard_qk[1] is not None:
        discard_qk[1] = discard_qk[1].view(num_heads, -1, 1).repeat(1, 1, 2).flatten()

    discard_qk[0] = prune_head_dim(qk_zero_out[0].flatten(), num_heads, layer.layer_id, 'QK')
    if discard_qk[0] is None and discard_qk[1] is None:
        return

    if discard_qk[0] is None:
        discard_qk[0] = torch.zeros_like(discard_qk[1]).bool()
    if discard_qk[1] is None:
        discard_qk[1] = torch.zeros_like(discard_qk[0]).bool()

    discard_qk = torch.cat([v.view(-1, qk_proj.split_sizes[i + 1]) for i, v in enumerate(discard_qk)], dim=-1).flatten()
    if qkvo_names[0] in discard_out:
        discard_qk = discard_qk[~discard_out[qkvo_names[0]]]

    def _select(output, select_index=None):
        return rearrange(
            rearrange(output, "... group hpg kv_len dim -> ... kv_len (group hpg dim)")[..., select_index],
            "... kv_len (group hpg dim2) -> ... group hpg kv_len dim2",
            hpg=output.shape[-3], group=output.shape[-4]
        )

    if qk_proj is not None and qk_proj.partial_factor is not None:
        if not hasattr(qk_proj, 'select_index'):
            qk_proj.select_index = {layer.layer_id: ~discard_qk}

            def qk_select(module, args, output):
                if module.layer_id in module.select_index:
                    output = (_select(output[0], module.select_index[module.layer_id]),
                              _select(output[1], module.select_index[module.layer_id]))
                return output

            qk_proj.register_forward_hook(qk_select)
        else:
            qk_proj.select_index[layer.layer_id] = ~discard_qk
    elif qk_norm_names is not None:
        for _name in qk_norm_names:
            layer.get_submodule(_name).register_forward_hook(
                lambda module, args, output: _select(output, ~discard_qk)
            )
    else:
        if qkvo_names[0] in discard_out:
            v_zero = discard_out[qkvo_names[0]].clone()
            v_zero[~v_zero] = discard_qk
            discard_qk = v_zero
        discard_out[qkvo_names[0]] = discard_out[qkvo_names[1]] = discard_qk

def rms_norm_forward(self, x):
    output = x * torch.rsqrt(
        x.pow(2).sum(dim=self.mean_dim, keepdim=True) / self.normalized_shape[0] + self.eps
    )
    if self.weight is not None:
        return output * self.weight
    return output


def forward(
    self,
    x,
    attn_mask = None,
    var_id = None,
    time_id = None,
    centroid = None,
):
    if self.pre_norm:
        if hasattr(self.self_attn, 'out_proj'):
            if (output_index := getattr(self.self_attn.out_proj, 'output_index', None)) is not None:
                x = x.index_add(-1, output_index, self._sa_block(self.norm1(x), attn_mask, var_id=var_id, time_id=time_id))
            else:
                x = x + self._sa_block(self.norm1(x), attn_mask, var_id=var_id, time_id=time_id)
        else:
            x = self.norm1(x)
        if hasattr(self.ffn, 'fc2'):
            if (output_index := getattr(self.ffn.fc2, 'output_index', None)) is not None:
                x = x.index_add(-1, output_index, self.ffn(self.norm2(x), centroid=centroid))
            else:
                x = x + self.ffn(self.norm2(x), centroid=centroid)
        else:
            x = self.norm2(x)
    else:
        if hasattr(self.self_attn, 'out_proj'):
            if (output_index := getattr(self.self_attn.out_proj, 'output_index', None)) is not None:
                x = self.norm1(x.index_add(-1, output_index, self._sa_block(x, attn_mask, var_id=var_id, time_id=time_id)))
            else:
                x = self.norm1(x + self._sa_block(x, attn_mask, var_id=var_id, time_id=time_id))
        else:
            x = self.norm1(x)
        if (output_index := getattr(self.ffn.fc2, 'output_index', None)) is not None:
            x = self.norm2(x.index_add(-1, output_index, self.ffn(x, centroid=centroid)))
        else:
            x = self.norm2(x + self.ffn(x, centroid=centroid))
    return x