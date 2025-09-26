import typing

import torch
from layers.prune_mask import MaskedLayer
from torch import nn

class Zero(nn.Module):
    def __init__(self, out_dim, num_attn_outputs: int = 1):
        super().__init__()
        self.out_dim = out_dim
        self.extra_attn_outputs = num_attn_outputs - 1

    def forward(self, x=None, *args, **kwargs):
        if x is None:
            x = args[0] if len(args) else list(kwargs.values())[0]
        if self.extra_attn_outputs <= 0:
            return x.new_zeros([1])
        else:
            return x.new_zeros([self.out_dim]), *([None] * self.extra_attn_outputs)

class Bias(nn.Module):
    def __init__(self, bias: nn.Parameter, num_attn_outputs: int = 1):
        super().__init__()
        self.bias = nn.Parameter(bias.data)
        self.extra_attn_outputs = num_attn_outputs - 1

    def forward(self, x=None, *args, **kwargs):
        if self.extra_attn_outputs <= 0:
            return self.bias
        else:
            return self.bias, *([None] * self.extra_attn_outputs)


def merge_weights(layer: nn.Module, names: list[str] = None, qkvo_names: list[str] = None,
                  dependency_graph: dict[str, tuple[dict, dict]] = None,
                  num_heads=None, revise_head_num: typing.Callable = None,
                  prune_self_attention: typing.Callable = None, enable_index_add: bool = False,
                  reduce_V: bool = True, num_attn_outputs: int = 1):
    if isinstance(layer, MaskedLayer):
        layer.merge_weight_()
        layer.merged = True
    elif isinstance(layer, nn.ModuleList):
        for i, _layer in enumerate(layer):
            _layer.layer_id = i
            merge_weights(_layer, names, qkvo_names, dependency_graph,
                          num_heads, revise_head_num, enable_index_add=enable_index_add,
                          prune_self_attention=prune_self_attention,
                          reduce_V=reduce_V, num_attn_outputs=num_attn_outputs)
    else:
        def merge_mask(mask, name, dependency_idx, history_names):
            history_names.append(name)
            if name in dependency_graph:
                for name2, dim in dependency_graph[name][dependency_idx].items():
                    if name2 in history_names:
                        continue
                    history_names.append(name2)
                    affected = layer.get_submodule(name2)
                    if hasattr(affected, 'mask_out'):
                        mask |= (affected.mask_out.mask == 0) if dim == 1 else (affected.mask_in.mask == 0)
                    else:
                        mask |= (affected.weight == 0) if affected.weight.data.ndim == 1 else (
                            (affected.weight.abs().sum(dim) == 0))
                    mask |= merge_mask(mask, name2, 1 - dim, history_names)
            return mask

        discard_in, discard_out = {}, {}

        for name in names:
            try:
                module = layer.get_submodule(name)
            except:
                continue
            if not isinstance(module, MaskedLayer):
                zero_in = (module.weight == 0) if module.weight.data.ndim == 1 else (module.weight.abs().sum(0) == 0)
                zero_out = (module.weight == 0) if module.weight.data.ndim == 1 else (module.weight.abs().sum(1) == 0)
            else:
                zero_in, zero_out = module.mask_in.mask == 0, module.mask_out.mask == 0
            zero_in = merge_mask(zero_in, name, 0, [])
            zero_out = merge_mask(zero_out, name, 1, [])
            if hasattr(module, 'mask_out'):
                module.mask_out.mask.data.masked_fill_(zero_out, 0)
            if hasattr(module, 'mask_in'):
                module.mask_in.mask.data.masked_fill_(zero_in, 0)
            if name in dependency_graph:
                if name not in qkvo_names:
                    if len(dependency_graph[name][0]):
                        discard_in[name] = zero_in
                    if len(dependency_graph[name][1]):
                        discard_out[name] = zero_out
            if qkvo_names and name == qkvo_names[2]:
                # zero value head -> zero qkv head
                num_heads = num_heads or zero_in // 64
                _num_heads = num_heads
                head_dim = zero_out.shape[-1] // num_heads
                discard_head = (~zero_out).view(num_heads, -1).sum(-1, keepdim=True) == 0
                if discard_head.any():
                    print(f"Layer {layer.layer_id}: Reduce head number by {discard_head.sum().item()} / {num_heads}")
                    revise_head_num(layer, discard_head)
                    num_heads -= discard_head.sum()
                    discard_head = discard_head.repeat(1, head_dim).reshape(-1)
                    for name2 in qkvo_names[:3]:
                        layer.get_submodule(name2).mask_out.mask.data.masked_fill_(discard_head, 0)
                        discard_out[name2] = discard_head.clone()
                    # zero_out = zero_out[~discard_head]
                # discard_v = (~zero_out).view(num_heads, -1).sum(0, keepdim=True) == 0
                # print(f"Layer {layer.layer_id}: Reduce V head dim by {discard_v.sum().item()} / {head_dim}")
                # discard_out[name] = discard_v.repeat(num_heads, 1).reshape(-1)
                if num_heads > 0 and reduce_V:
                    discard_v = prune_head_dim(zero_out, _num_heads, layer.layer_id, 'V')
                    if discard_v is not None:
                        # if qkvo_names[2] in discard_out:
                        #     v_zero = discard_out[qkvo_names[0]].clone()
                        #     v_zero[~v_zero] = discard_v
                        #     discard_v = v_zero
                        if qkvo_names[2] in discard_out:
                            discard_out[qkvo_names[2]] |= discard_v
                        else:
                            discard_out[qkvo_names[2]] = discard_v

        if qkvo_names and qkvo_names[2] in discard_out and len(qkvo_names) == 4:
            discard_in[qkvo_names[3]] = discard_out[qkvo_names[2]]
            layer.get_submodule(qkvo_names[3]).mask_in.mask.data.masked_fill_(discard_out[qkvo_names[2]], 0)

        # if num_heads == 0:
        #     if isinstance(bias := layer.get_submodule(qkvo_names[-1]).bias, nn.Parameter):
        #         layer.register_module('.'.join(qkvo_names[0].split('.')[:-1]), Bias(bias))
        #     else:
        #         layer.register_module('.'.join(qkvo_names[0].split('.')[:-1]), Zero())
        #     names = [name for name in names if name not in qkvo_names]
        if prune_self_attention is not None:
            prune_self_attention(layer=layer, qkvo_names=qkvo_names, discard_out=discard_out,
                                 num_heads=num_heads, head_dim=head_dim,)

        for name in list(names):
            try:
                module = layer.get_submodule(name)
            except:
                continue
            is_outer = name == names[-1]
            if len(qkvo_names) > 3: is_outer = is_outer or name == qkvo_names[-1]
            if isinstance(module, MaskedLayer):
                # if is_outer and name in discard_out and not (discard_out.get(name) == 0).any():
                #     TODO: handle multiple outputs of attention blocks, e.g., in Time-MoE.
                if (name in discard_out and not (discard_out[name] == 0).any() or
                        name in discard_in and not (discard_in[name] == 0).any() or
                        module.mask_out.mask.sum() == 0 or module.mask_in.mask.sum() == 0):
                    print('All zero outputs:', name)
                    out_dim = len(discard_out[name]) if name in discard_out else len(module.mask_out.mask)
                    parent = layer.get_submodule('.'.join(name.split('.')[:-2 if is_outer else -1]))
                    module_name = name[-2 if is_outer else -1]
                    if isinstance(bias := layer.get_submodule(name).bias, nn.Parameter):
                        parent.register_module(module_name, Bias(bias, num_attn_outputs))
                    else:
                        parent.register_module(module_name, Zero(out_dim, num_attn_outputs))
                else:
                    module.merge_weight_(discard_in.get(name), discard_out.get(name),
                                         is_outer=enable_index_add and is_outer)
            elif hasattr(module, 'weight'):
                if name in discard_in:
                    module.weight.data = module.weight.data[..., ~discard_in[name]]
                if name in discard_out:
                    module.weight.data = module.weight.data[~discard_out[name]]
            module.merged = True


def prune_head_dim(zero_out: torch.BoolTensor, num_heads: int, layer_id: int, name: str):
    head_zero_out = zero_out.view(num_heads, -1)
    num_zero_per_head = head_zero_out.sum(-1).min()
    if (num_zero_per_head > 0).any():
        head_dim = head_zero_out.shape[-1]
        print(f"Layer {layer_id}: Reduce {name} head dim by {num_zero_per_head.item()} / {head_dim}")
        discard_out = torch.cat([
            torch.arange(head_dim, device=head_zero_out.device)[head_zero_out[i]][
            -num_zero_per_head:] + head_dim * i
            for i in range(num_heads)
        ])
        discard_out = torch.zeros_like(zero_out).index_fill(-1, discard_out, 1).bool()
        return discard_out
    return None