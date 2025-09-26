
from functools import cache
from typing import Callable

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from utils.tools import as_buffer_
from layers.lora import add_lora_

def add_masks_(root_module: nn.Module, exclude_name=None, valid_name_fn: Callable = None, **kwargs):
    root_module.eval()
    root_module.requires_grad_(False)
    if kwargs.get('prune_transformer', False) and hasattr(root_module, 'transformers'):
        kwargs['prune_transformer'] = False
        for module in root_module.transformers:
            add_masks_(module, exclude_name=exclude_name, valid_name_fn=valid_name_fn, **kwargs)
    else:
        for name, module in root_module.named_modules():
            if (isinstance(module, (nn.Conv1d, nn.Linear, nn.LayerNorm))
                    and isinstance(module.weight, nn.Parameter) and not isinstance(module, MaskedLayer)
                    and (exclude_name is None or name.split('.')[-1] not in exclude_name)
                    and (valid_name_fn is None or valid_name_fn(name))):
                add_mask_(root_module, name, load_weights=True, merge_weights=True,
                          lora_rank=kwargs.get('lora_rank'), lora_alpha=kwargs.get('lora_alpha'))
    if kwargs.get('lora_rank'):
        for name, module in root_module.named_modules():
            if isinstance(module, nn.Linear) and not isinstance(module, MaskedLayer):
                parent_module = root_module.get_submodule('.'.join(name.split('.')[:-1])) if '.' in name else root_module
                add_lora_(parent_module, name.split('.')[-1], r=kwargs.get('lora_rank'), lora_alpha=kwargs.get('lora_alpha'))
    # for param in root_module.parameters():
    #     if param.ndim == 1:
    #         param.requires_grad = True
    return root_module

def add_mask_(root_module: nn.Module, module_name: str, load_weights=True, merge_weights=True, **kwargs):
    old_module = root_module.get_submodule(module_name)
    parent_module = root_module.get_submodule('.'.join(module_name.split('.')[:-1])) if '.' in module_name else root_module
    if isinstance(old_module, nn.Linear):
        if kwargs.get('lora_rank'):
            new_module = LoRALinear(in_features=old_module.in_features, out_features=old_module.out_features,
                                bias=old_module.bias is not None,
                                device=old_module.weight.device, dtype=old_module.weight.dtype,
                                merge_weights=merge_weights, **kwargs)
        else:
            new_module = Linear(in_features=old_module.in_features, out_features=old_module.out_features,
                                bias=old_module.bias is not None,
                                device=old_module.weight.device, dtype=old_module.weight.dtype,
                                merge_weights=merge_weights, **kwargs)
    else:
        return
    if load_weights:
        new_module.load_state_dict(old_module.state_dict(), strict=False)
    new_module._forward_hooks = old_module._forward_hooks
    new_module._forward_pre_hooks = old_module._forward_pre_hooks
    setattr(parent_module, module_name.split('.')[-1], new_module)

@cache
def zero_tensor(num_batch, dim, device, dtype):
    return torch.zeros(num_batch, dim, device=device, dtype=dtype)

@torch.no_grad()
def accumulate_fisher(module, grad_out):
    if module.track_grad:
        grad = grad_out[0]
        if hasattr(module, 'batch_id') and len(grad) > 0:
            sample_cnt = len(torch.unique(module.batch_id))
            module.sample_cnt += sample_cnt
        if module.track_grad != 'taylor2':
            return
        if hasattr(module, 'batch_id') and len(grad) > 0:
            sample_grad = zero_tensor(module.batch_id.max() + 1, grad.shape[-1], device=grad.device, dtype=grad.dtype)
            sample_grad = sample_grad.index_add(0, module.batch_id, grad)
            score = sample_grad - sample_grad ** 2 / 2
            score = score.sum(0, keepdim=True)
            score = (score / grad.shape[0]).expand(grad.shape[0], -1) # must recast to the original shape
            return (score, )
        else:
            score = grad - grad ** 2 / 2
        return (score, )

class Mask(nn.Module):
    def __init__(self, dim, newaxis=0, track_grad: bool = True, dtype=None, device=None):
        super().__init__()
        self.dim = dim
        self.track_grad = track_grad
        self.mask = nn.Parameter(torch.ones((dim, ) + (1, ) * newaxis, dtype=dtype, device=device))
        self.token_cnt = 0
        self.sample_cnt = 0
        self.register_full_backward_pre_hook(accumulate_fisher)

    def forward(self, x, batch_size=None):
        if not self.mask.requires_grad:
            return self.mask
        self.token_cnt += x.size(0)
        if batch_size is None:
            batch_size = [x.size(i) for i in range(max(1, x.ndim-2))]
        return self.mask.expand(*batch_size, -1)

    def read_use_count(self, multi_gpu=False):
        if multi_gpu:
            self.token_cnt = torch.tensor(self.token_cnt, device=self.mask.device)
            self.sample_cnt = torch.tensor(self.sample_cnt, device=self.mask.device)
            dist.all_reduce(self.token_cnt, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.sample_cnt, op=dist.ReduceOp.SUM)
            sample_cnt, token_cnt = self.sample_cnt.item(), self.token_cnt.item()
        else:
            sample_cnt, token_cnt = self.sample_cnt, self.token_cnt
        self.sample_cnt = self.token_cnt = 0
        return sample_cnt, token_cnt


class MaskedLayer(object):
    def __init__(self, in_features: int, out_features: int, track_grad: bool = True, freeze_bias: bool = True,
                 merge_weights: bool = True, newaxis: int =0, dtype: torch.dtype = None, device=None, **kwargs) -> None:
        assert isinstance(self, nn.Module)
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.track_grad = track_grad
        self.dtype = dtype

        # Actual trainable parameters
        self.mask_out = Mask(out_features, newaxis, track_grad, dtype, device)
        self.mask_in = Mask(in_features, newaxis, track_grad, dtype, device)

        self.weight.requires_grad = False
        if getattr(self, 'bias', None) is not None and freeze_bias:
            self.bias.requires_grad = False

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        if self.mask_out is None and self.mask_in is None:
            return
        if mode:
            if self.merge_weights and self.merged:
                self.restore()
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                weight, bias = self._merge(self.weight.data, self.bias if hasattr(self, "bias") and
                                                                          not isinstance(self.bias, bool) else None)
                self.weight.data = weight.detach()
                if bias is not None:
                    self.bias.data = bias.detach()
                self.merged = True

    def _merge(self, weight, bias):
        if bias is not None and self.mask_out is not None:
            bias = self.mask_out.mask.squeeze() * bias
        if weight is not None:
            mask = 1
            if self.mask_in is not None:
                mask = mask * self.mask_in.mask
            if self.mask_out is not None:
                mask = mask * self.mask_out.mask.unsqueeze(-1)
            weight = weight * mask
        return weight, bias

    def restore(self):
        # pruned weights are not restorable
        pass

    def merge_weight_(self, discard_in = None, discard_out = None, is_outer: bool = False):
        self.track_grad = False
        self.merge_weights = True
        # self.train(False)
        weight, bias = self._merge(self.weight.data, self.bias if hasattr(self, "bias") and
                                                                          not isinstance(self.bias, bool) else None)
        zero_in = (weight.abs().sum(0) == 0).squeeze()
        zero_out = (weight.abs().sum(1) == 0).squeeze()
        if bias is not None:
            zero_out &= bias == 0

        as_buffer_(self.mask_in, 'mask')
        as_buffer_(self.mask_out, 'mask')

        if discard_in is not None:
            weight = weight[:, ~discard_in]
            if zero_in.sum() == discard_in.sum():
                self.mask_in = None
            else:
                self.mask_in.mask = self.mask_in.mask[~discard_in] > 0

        if is_outer:
            self.register_buffer('output_index', None, persistent=True)
        if zero_out.any():
            if is_outer or discard_out is not None:
                if is_outer:
                    nonzero_mask = ~zero_out
                    self.output_index = torch.arange(len(nonzero_mask), dtype=torch.int, device=zero_out.device)[nonzero_mask]
                else:
                    nonzero_mask = ~discard_out

                weight = weight[nonzero_mask]
                if bias is not None:
                    bias = bias[nonzero_mask]
                if is_outer or zero_out.sum() == discard_out.sum():
                    self.mask_out = None
                else:
                    self.mask_out.mask = self.mask_out.mask[nonzero_mask] > 0
        self.weight.data = weight
        if bias is not None:
            self.bias.data = bias
        self.out_features = self.weight.data.shape[0]
        self.in_features = self.weight.data.shape[1]
        self.merged = True


class Linear(nn.Linear, MaskedLayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None, dtype=None,
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias, device=device, dtype=dtype)
        MaskedLayer.__init__(self, in_features, out_features=out_features, merge_weights=merge_weights, newaxis=0,
                             dtype=dtype, device=device, **kwargs)

    def train(self, mode: bool = True):
        MaskedLayer.train(self, mode)

    def forward(self, x: torch.Tensor):
        if self.track_grad and self.training:
            batch_size = [x.size(i) for i in range(max(1, x.ndim-2))]
            mask_in = self.mask_in(x, batch_size)
            if x.ndim > 2:
                mask_in = mask_in.unsqueeze(-2)
            x = x * mask_in
            res = F.linear(x, self.weight, bias=self.bias)
            mask_out = self.mask_out(res, batch_size)
            if res.ndim > 2:
                mask_out = mask_out.unsqueeze(-2)
            return res * mask_out
        elif not self.merged:
            weight, bias = self._merge(self.weight, self.bias)
            res = F.linear(x, weight, bias=bias)
        else:
            res = F.linear(x, self.weight, bias=self.bias)
        return res


class LoRALinear(Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None, dtype=None,
            merge_weights: bool = True,
            lora_rank: int = 0,
            lora_alpha: int = 1,
            fan_in_fan_out: bool = False,
            **kwargs
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype, merge_weights=merge_weights,
                         **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self.r = lora_rank
        self.merged_lora = False
        if lora_rank > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((lora_rank, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, lora_rank)))
            self.scaling = lora_alpha / self.r
            self.weight.requires_grad = False
            if getattr(self, 'bias', None) is not None:
                self.bias.requires_grad = True
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if mode:
            if self.merge_weights and self.merged_lora:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged_lora = False
        else:
            if self.merge_weights and not self.merged_lora:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged_lora = True
        super().train(mode)

    def _merge(self, weight, bias):
        if not self.track_grad and not self.merged_lora:
            weight = weight + self.lora_B @ self.lora_A * self.scaling
        return super()._merge(weight, bias)

    def merge_weight_(self, discard_in = None, discard_out = None, is_outer: bool = False):
        self.track_grad = False
        self.merge_weights = True
        # self.train(False)
        weight, bias = self._merge(self.weight.data, self.bias if hasattr(self, "bias") and
                                                                          not isinstance(self.bias, bool) else None)
        zero_in = (weight.abs().sum(0) == 0).squeeze()
        zero_out = (weight.abs().sum(1) == 0).squeeze()
        if bias is not None:
            zero_out &= bias == 0

        as_buffer_(self.mask_in, 'mask')
        as_buffer_(self.mask_out, 'mask')

        if discard_in is not None:
            weight = weight[:, ~discard_in]
            self.lora_A.data = self.lora_A[:, ~discard_in]
            if zero_in.sum() == discard_in.sum():
                self.mask_in = None
            else:
                self.mask_in.mask = self.mask_in.mask[~discard_in] > 0

        if is_outer:
            self.register_buffer('output_index', None, persistent=True)
        if zero_out.any():
            if is_outer or discard_out is not None:
                if is_outer:
                    nonzero_mask = ~zero_out
                    self.output_index = torch.arange(len(nonzero_mask), dtype=torch.int, device=zero_out.device)[nonzero_mask]
                else:
                    nonzero_mask = ~discard_out

                weight = weight[nonzero_mask]
                self.lora_B.data = self.lora_B[nonzero_mask]
                if bias is not None:
                    bias = bias[nonzero_mask]
                if is_outer or zero_out.sum() == discard_out.sum():
                    self.mask_out = None
                else:
                    self.mask_out.mask = self.mask_out.mask[nonzero_mask] > 0
        self.weight.data = weight
        if bias is not None:
            self.bias.data = bias
        self.out_features = self.weight.data.shape[0]
        self.in_features = self.weight.data.shape[1]
        self.merged = True
        # if hasattr(self, 'lora_A'):
        #     nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
