import sys
import types
from functools import partial, cache
from typing import Optional, List, Tuple

import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM

from layers.prune_mask import MaskedLayer
from models.base import PrunableModel
from utils.pruning import prune_head_dim, merge_weights


class Model(nn.Module, PrunableModel):
    def __init__(self, args):
        super(Model, self).__init__()
        size = '200M' if args.model_size == 'large' else '50M'
        self.dtype = torch.bfloat16
        self.attn_implementation = getattr(args, 'attn_implementation', 'flash_attention_2')
        self.model = AutoModelForCausalLM.from_pretrained(
            f'Maple728/TimeMoE-{size}',
            torch_dtype = 'auto',
            attn_implementation = self.attn_implementation,
            device_map="cpu",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
            trust_remote_code=True,
        )
        self.model.apply_aux_loss = self.model.config.apply_aux_loss = getattr(args, "apply_aux_loss", True)
        self.pred_len = args.pred_len
        self.revin = args.revin
        self.batch_id_handler = None
        self.num_experts_per_tok = self.model.num_experts_per_tok
        for layer in self.transformers:
            layer.ffn_layer.register_buffer("pruned_expert_ids", None, persistent=True)
        self.remove_batch_id_handler_()
        self.transformer_names = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                                  'gate_proj', 'up_proj', 'down_proj']

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
            return_dict: Optional[bool] = None,
            max_horizon_length: Optional[int] = None,):
        if self.revin:
            mean, std = input_ids.mean(dim=-2, keepdim=True), input_ids.std(dim=-2, keepdim=True) + 1e-6
            input_ids = (input_ids - mean) / std
        if self.training or labels is not None:
            if not self.training:
                apply_aux_loss, self.model.apply_aux_loss = self.model.apply_aux_loss, False
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
                max_horizon_length=max_horizon_length,
            )
            if not self.training:
                self.model.apply_aux_loss = apply_aux_loss
            if labels is not None:
                predictions = outputs.logits
                if self.revin:
                    predictions = predictions * std + mean
                return predictions, outputs.loss
            else:
                predictions = outputs.logits
        else:
            predictions = self.model.generate(input_ids.squeeze(-1),
                                              max_new_tokens=self.pred_len,
                                              attention_mask=attention_mask,
                                              use_cache=use_cache)[:, -self.pred_len:].unsqueeze(-1)
        if self.revin:
            predictions = predictions * std + mean
        return predictions

    @property
    def transformers(self):
        return self.model.model.layers

    @property
    def experts(self):
        return [layer.ffn_layer.experts for layer in self.transformers]

    @property
    def ffns(self):
        return [layer.ffn_layer for layer in self.transformers]

    def merge_weights_(self):
        dependency_graph = {
            'self_attn.v_proj': ({}, {'self_attn.o_proj': 0}),
            'self_attn.o_proj': ({'self_attn.v_proj': 1}, {}),
        } # 0 is input dimension, 1 is output dimension
        qkvo_names = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
        names = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
        for i in range(len(self.experts[0])):
            prefix = f'ffn_layer.experts.{i}.'
            dependency_graph.update(**{
                prefix + 'up_proj': ({}, {prefix + 'gate_proj': 1, prefix + 'down_proj': 0}),
                prefix + 'gate_proj': ({}, {prefix + 'up_proj': 1, prefix + 'down_proj': 0}),
                prefix + 'down_proj': ({prefix + 'gate_proj': 1, prefix + 'up_proj': 1}, {}),
            })
            names += [prefix + 'gate_proj', prefix + 'up_proj', prefix + 'down_proj']
        prefix = f'ffn_layer.shared_expert.'
        dependency_graph.update(**{
            prefix + 'up_proj': ({}, {prefix + 'gate_proj': 1, prefix + 'down_proj': 0}),
            prefix + 'gate_proj': ({}, {prefix + 'up_proj': 1, prefix + 'down_proj': 0}),
            prefix + 'down_proj': ({prefix + 'gate_proj': 1, prefix + 'up_proj': 1}, {}),
        })
        names += [prefix + 'gate_proj', prefix + 'up_proj', prefix + 'down_proj']
        def revise_head_num(layer, discard_head):
            layer.self_attn.num_heads -= discard_head.sum()
            layer.self_attn.num_key_value_heads -= discard_head.sum()

        for k, module in sys.modules.items():
            if 'modeling_time_moe' in k:
                module.__dict__['TimeMoeDecoderLayer'].forward = forward
                if self.attn_implementation == 'eager':
                    module.__dict__['apply_rotary_pos_emb'] = apply_rotary_pos_emb

        merge_weights(self.transformers,
                      names=names,
                      qkvo_names=qkvo_names,
                      dependency_graph=dependency_graph,
                      num_heads=self.transformers[0].self_attn.num_heads,
                      revise_head_num=revise_head_num, enable_index_add=False,
                      prune_self_attention=None if self.attn_implementation != 'eager'
                      else partial(prune_self_attention, num_heads=self.transformers[0].self_attn.num_heads,)
                      )
        for layer in self.transformers:
            if getattr(layer.ffn_layer, 'pruned_expert_ids', None) is not None:
                ids = layer.ffn_layer.pruned_expert_ids
                if ids.dtype == torch.bool:
                    layer.ffn_layer.pruned_expert_ids = torch.arange(ids, dtype=torch.int64, device=ids.device)[ids]
                for j in layer.ffn_layer.pruned_expert_ids:
                    layer.ffn_layer.experts[j] = nn.Identity()

        for module in self.model.modules():
            if (module not in self.transformers and isinstance(module, MaskedLayer)
                    and not (hasattr(module, 'merged') and module.merged)):
                merge_weights(module)

    def register_batch_id_handler_(self):
        for layer in self.transformers:
            layer.ffn_layer.track_batch_id = True
            layer.ffn_layer.forward = types.MethodType(moe_forward, layer.ffn_layer)

    def remove_batch_id_handler_(self):
        for layer in self.transformers:
            layer.ffn_layer.track_batch_id = False

@cache
def get_batch_ids(batch_size, sequence_length, device):
    return torch.arange(batch_size, device=device).unsqueeze(-1).expand(-1, sequence_length).flatten()

def moe_forward(self, hidden_states: torch.Tensor):
    """ """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits -> (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    """ New feature: Masked pruned experts """
    if getattr(self, 'pruned_expert_ids', None) is not None:
        routing_weights = routing_weights.index_fill(-1, self.pruned_expert_ids, 0)

    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

    # """ New feature: Labeling the batch id on each token """
    # if self.training and getattr(self, 'track_batch_id', False):
    #     batch_id = get_batch_ids(batch_size, sequence_length, device=selected_experts.device)
    #     track_batch_id(batch_id, self.experts, selected_experts)

    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])

        """ New feature: Labeling the batch id on each token """
        if self.training and getattr(self, 'track_batch_id', False) and not isinstance(expert_layer, nn.Identity):
            _batch_id = top_x // sequence_length
            for submodule in expert_layer.modules():
                if isinstance(submodule, MaskedLayer):
                    submodule.mask_in.batch_id = _batch_id
                    submodule.mask_out.batch_id = _batch_id

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        if current_hidden_states.shape[-1] == final_hidden_states.shape[-1]:
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        else:
            final_hidden_states[top_x] = final_hidden_states[top_x].index_add_(-1,
                                                                               self.experts[expert_idx].down_proj.output_index,
                                                  current_hidden_states.to(hidden_states.dtype))

    shared_expert_output = self.shared_expert(hidden_states)
    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

    final_hidden_states = final_hidden_states + shared_expert_output

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits

@torch.no_grad()
def track_batch_id(batch_id, experts, selected_experts):
    for expert_id in range(len(experts)):
        _batch_id = batch_id.masked_select((selected_experts == expert_id).max(-1)[0])
        for submodule in experts[expert_id].modules():
            if isinstance(submodule, MaskedLayer):
                submodule.mask_in.batch_id = _batch_id
                submodule.mask_out.batch_id = _batch_id


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

    hidden_states = self.input_layernorm(hidden_states)

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

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states, router_logits = self.ffn_layer(hidden_states)
    hidden_states = residual + hidden_states

    if not output_attentions:
        self_attn_weights = None

    if not use_cache:
        present_key_value = None
    return hidden_states, self_attn_weights, present_key_value, router_logits


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
