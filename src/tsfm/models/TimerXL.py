import math
import sys
from functools import partial
from typing import Optional, List, Tuple
import torch
from torch import nn
from transformers import AutoModelForCausalLM

from layers.prune_mask import MaskedLayer
from models.base import PrunableModel
from utils.pruning import merge_weights, prune_head_dim

class Model(nn.Module, PrunableModel):
    def __init__(self, args):
        super(Model, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            'thuml/timer-base-84m',
            trust_remote_code=True,
            # torch_dtype='auto',
            # attn_implementation='flash_attention_2',
            # device_map="auto",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
        )
        self.model.config.output_token_lens = [96]
        self.revin = True
        self.pred_len = args.pred_len
        self.num_heads = self.transformers[0].self_attn.num_heads
        self.transformer_names = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                                  'ffn_layer.gate_proj', 'ffn_layer.up_proj', 'ffn_layer.down_proj']

    def forward(self,
            input_ids: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.FloatTensor] = None,
            loss_masks: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,):
        input_ids = input_ids.squeeze(-1)
        if self.revin:
            mean, std = input_ids.mean(dim=-1, keepdim=True), input_ids.std(dim=-1, keepdim=True) + 1e-6
            input_ids = (input_ids - mean) / std
        if self.training:
            if labels is not None:
                labels = labels.squeeze(-1)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                loss_masks=loss_masks,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                revin=False,
            )
            if labels is not None:
                predictions = outputs.logits
                return predictions, outputs.loss
            else:
                predictions = outputs.logits
        else:
            predictions = self.model.generate(input_ids,
                                              max_new_tokens=self.pred_len,
                                              attention_mask=attention_mask,
                                              use_cache=use_cache,
                                              revin=False,)
        if self.revin:
            predictions = predictions * std + mean
        return predictions.unsqueeze(-1)

    @property
    def transformers(self):
        return self.model.model.layers


    def merge_weights_(self):
        dependency_graph = {
            'self_attn.v_proj': ({}, {'self_attn.o_proj': 0}),
            'self_attn.o_proj': ({'self_attn.v_proj': 1}, {}),
            'ffn_layer.up_proj': ({}, {'ffn_layer.gate_proj': 1, 'ffn_layer.down_proj': 0}),
            'ffn_layer.gate_proj': ({}, {'ffn_layer.up_proj': 1, 'ffn_layer.down_proj': 0}),
            'ffn_layer.down_proj': ({'ffn_layer.gate_proj': 1, 'ffn_layer.up_proj': 1}, {}),
        } # 0 is input dimension, 1 is output dimension
        qkvo_names = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']

        def revise_head_num(layer, discard_head):
            layer.self_attn.num_heads -= discard_head.sum()

        # torch.index_add is not friendly to CPU
        enable_index_add = False
        for k, module in sys.modules.items():
            if 'modeling_timer' in k:
                module.__dict__['apply_rotary_pos_emb'] = apply_rotary_pos_emb
                if enable_index_add:
                    module.__dict__['TimerDecoderLayer'].forward = forward

        merge_weights(self.transformers,
                      names=self.transformer_names,
                      qkvo_names=qkvo_names,
                      dependency_graph=dependency_graph,
                      num_heads=self.transformers[0].self_attn.num_heads,
                      revise_head_num=revise_head_num, enable_index_add=enable_index_add,
                      prune_self_attention=partial(prune_self_attention,
                                                   num_heads=self.transformers[0].self_attn.num_heads,)
                      )
        for layer in self.transformers:
            layer.self_attn.head_dim = -1
            layer.self_attn.hidden_size = -1

        for module in self.model.modules():
            if (module not in self.transformers and isinstance(module, MaskedLayer)
                    and not (hasattr(module, 'merged') and module.merged)):
                merge_weights(module)


def prune_self_attention(layer, discard_out, qkvo_names, head_dim, num_heads):
    q_zero_out = layer.get_submodule(qkvo_names[0]).mask_out.mask == 0
    k_zero_out = layer.get_submodule(qkvo_names[1]).mask_out.mask == 0
    qk_zero_out = torch.stack([q_zero_out, k_zero_out], dim=0).view(2, -1, head_dim)

    qk_zero_out[..., :qk_zero_out.shape[-1] // 2] = \
    qk_zero_out[..., qk_zero_out.shape[-1] // 2:] = \
    qk_zero_out[..., :qk_zero_out.shape[-1] // 2] & qk_zero_out[..., qk_zero_out.shape[-1] // 2:]

    qk_zero_out = (qk_zero_out[0] | qk_zero_out[1]).flatten()
    if qkvo_names[0] in discard_out:
        qk_zero_out = qk_zero_out[~discard_out[qkvo_names[0]]]
    qk_zero_out = qk_zero_out.view(num_heads, -1)
    discard_qk = prune_head_dim(qk_zero_out[..., :qk_zero_out.shape[-1] // 2].flatten(), num_heads, layer.layer_id, 'QK')
    if discard_qk is not None:
        discard_qk = discard_qk.view(num_heads, -1).repeat(1, 2).flatten()
        if qkvo_names[0] in discard_out:
            # v_zero = discard_out[qkvo_names[0]].clone()
            # v_zero[~v_zero] = discard_qk
            # discard_qk = v_zero
            discard_qk |= discard_out[qkvo_names[0]]
        discard_out[qkvo_names[0]] = discard_out[qkvo_names[1]] = discard_qk

    layer.self_attn.rotary_emb.cos_cached = layer.self_attn.rotary_emb.cos_cached.unsqueeze(1)
    layer.self_attn.rotary_emb.sin_cached = layer.self_attn.rotary_emb.sin_cached.unsqueeze(1)

    if discard_qk is not None:
        layer.self_attn.rotary_emb.cos_cached = select_rope(layer.self_attn.rotary_emb.cos_cached, ~discard_qk, num_heads)
        layer.self_attn.rotary_emb.sin_cached = select_rope(layer.self_attn.rotary_emb.sin_cached, ~discard_qk, num_heads)

        # When the scaling factor in F.scaled_dot_product_attention is not defined, exp (qk/d) = exp (d'/d qk/d')
        rescaling = math.sqrt((~discard_qk).sum() // num_heads / head_dim)
        layer.get_submodule(qkvo_names[0]).weight.data *= rescaling

def select_rope(emb, mask, num_heads):
    return emb.repeat(1, num_heads, 1).view(emb.size(0), -1)[:, mask].reshape(emb.size(0), num_heads, -1)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].transpose(1, 2)
    sin = sin[position_ids].transpose(1, 2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
    residual = hidden_states

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    if (output_index := getattr(self.self_attn.o_proj, 'output_index', None)) is not None:
        hidden_states = residual.index_add(-1, output_index, hidden_states)
    else:
        hidden_states = residual + hidden_states
    hidden_states = self.norm1(hidden_states)

    # Fully Connected
    residual = hidden_states
    hidden_states = self.ffn_layer(hidden_states)
    if (output_index := getattr(self.ffn_layer.down_proj, 'output_index', None)) is not None:
        hidden_states = residual.index_add(-1, output_index, hidden_states)
    else:
        hidden_states = residual + hidden_states
    hidden_states = self.norm2(hidden_states)

    if not output_attentions:
        self_attn_weights = None

    if not use_cache:
        present_key_value = None
    return hidden_states, self_attn_weights, present_key_value