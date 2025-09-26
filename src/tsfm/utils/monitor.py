import importlib
import os
from collections import OrderedDict

import torch
from torch import distributed as dist
from einops import rearrange
from torch import nn

def get_module_name(model_name, model):
    if model_name == 'TimerXL':
        kwargs = dict(num_heads=model.transformers[0].self_attn.num_heads,
                      o_proj_name='self_attn.o_proj', fc2_name='ffn_layer.down_proj')
    elif model_name == 'TimeMoE':
        kwargs = dict(num_heads=model.transformers[0].self_attn.num_heads,
                      o_proj_name='self_attn.o_proj',
                      fc2_name=[f'ffn_layer.experts.{i}.down_proj' for i in range(len(model.experts[0]))], )
    elif model_name == 'moirai':
        kwargs = dict(num_heads=model.transformers[0].self_attn.num_heads,
                      o_proj_name='self_attn.out_proj',
                      fc2_name='ffn.fc2')
    elif model_name == 'Chronos':
        kwargs = dict(num_heads=model.transformers[0].layer[0].SelfAttention.n_heads,
                      o_proj_name=['layer.0.SelfAttention.o', 'layer.1.EncDecAttention.o'],
                      fc2_name=['layer.1.DenseReluDense.wo', 'layer.2.DenseReluDense.wo'])
    elif model_name == 'TTM':
        kwargs = dict(num_heads=0,
                      o_proj_name=[],
                      fc2_name=['fc2'])
    elif model_name == 'TimesFM':
        kwargs = dict(num_heads=model.transformers[0].self_attn.num_heads,
                      o_proj_name=['self_attn.o_proj'],
                      fc2_name=['mlp.down_proj'])
    elif model_name == 'PatchTST':
        kwargs = dict(num_heads=model.transformers[0].self_attn.n_heads,
                      o_proj_name=['self_attn.to_out.0'],
                      fc2_name=['ff.3'])
    else:
        raise NotImplementedError(model)
    if not isinstance(kwargs["o_proj_name"], list):
        kwargs["o_proj_name"] = [kwargs["o_proj_name"]]
    if not isinstance(kwargs["fc2_name"], list):
        kwargs["fc2_name"] = [kwargs["fc2_name"]]
    return kwargs

def wrap_model(root_model, **kwargs):
    kwargs = get_module_name(kwargs.get('model'), root_model)
    add_forward_hooks(getattr(root_model, "transformers", root_model), **kwargs)
    root_model.eval()
    return root_model

def add_forward_hooks(transformers: nn.ModuleList, num_heads,
                      o_proj_name: str = 'self_attn.o_proj',
                      fc2_name: str = 'ffn_layer.fc2',
                      ):
    if isinstance(transformers, (nn.ModuleList, list)):
        transformers[0].o_proj_name = o_proj_name
        transformers[0].fc2_name = fc2_name
        print(transformers[0].fc2_name)
        for i, transformer in enumerate(transformers):
            def save_attn_res_norms(transformer, args, kwargs):
                if getattr(transformer, "enable_hook", False):
                    for name in o_proj_name:
                        try:
                            if len(args):
                                transformer.get_submodule(name).res_norm = args[0].norm(dim=-1) # BS q_len 1
                            else:
                                transformer.get_submodule(name).res_norm = kwargs.get("hidden_states").norm(dim=-1)  # BS q_len 1
                        except Exception:
                            pass
            transformer.register_forward_pre_hook(save_attn_res_norms, with_kwargs=True)
            transformer.enable_hook = True

            def save_head_relative_norm(o_proj, args, outputs):
                if getattr(o_proj, "enable_hook", False):
                    value = args[0].reshape(args[0].size(0), -1, num_heads, args[0].size(-1) // num_heads).transpose(1, 2) # BS, H, q_len, vdim
                    head_w = rearrange(o_proj.weight, "dim (H vdim) -> H vdim dim", H=num_heads)
                    out_proj = (value @ head_w).transpose(-2, -3)  # BS q_len H dim
                    relative_norm = (out_proj.norm(dim=-1) / o_proj.res_norm.unsqueeze(-1)).reshape(-1, num_heads)
                    o_proj.head_norm_r += relative_norm.sum(0) # H
                    o_proj.attn_norm_r += (outputs.norm(dim=-1) / o_proj.res_norm).reshape(-1).sum(0)
                    o_proj.sparse_head_cnt += (relative_norm <= 0.05).sum(0) # H
                    o_proj.num_tokens += value.size(0) * value.size(2)
                    o_proj.res_norm = 0

            for name in o_proj_name:
                try:
                    o_proj = transformer.get_submodule(name)
                    o_proj.register_forward_hook(save_head_relative_norm)
                    o_proj.head_norm_r = 0
                    o_proj.attn_norm_r = 0
                    o_proj.sparse_head_cnt = 0
                    o_proj.num_tokens = 0
                    o_proj.enable_hook = True
                except Exception as e:
                    pass

    def save_activation_sum(fc2, args):
        if getattr(fc2, "enable_hook", False):
            act = args[0].view(-1, args[0].size(-1))
            fc2.act_value += act.abs().sum(0)
            fc2.act_num += (act > 0).sum(0)
            fc2.num_tokens += act.size(0)
    if isinstance(transformers, (nn.ModuleList, list)):
        for i, transformer in enumerate(transformers):
            for name in fc2_name:
                try:
                    fc2 = transformer.get_submodule(name)
                    fc2.register_forward_pre_hook(save_activation_sum)
                    fc2.act_value = 0
                    fc2.act_num = 0
                    fc2.num_tokens = 0
                    fc2.enable_hook = True
                except Exception as e:
                    pass
    else:
        transformers.fc2_name = fc2_name
        for module_name, fc2 in transformers.named_modules():
            if isinstance(fc2, nn.Linear) and module_name.split('.')[-1] in fc2_name:
                fc2.register_forward_pre_hook(save_activation_sum)
                fc2.act_value = 0
                fc2.act_num = 0
                fc2.num_tokens = 0
                fc2.enable_hook = True

@torch.no_grad()
def collect_states(transformers: nn.ModuleList, ):
    state_dict = OrderedDict()
    if isinstance(transformers, (nn.ModuleList, list)):
        o_proj_name = transformers[0].o_proj_name
        fc2_name = transformers[0].fc2_name
        print(transformers[0].fc2_name)
        for i, transformer in enumerate(transformers):
            for name in o_proj_name:
                try:
                    o_proj = transformer.get_submodule(name)
                    if int(os.getenv("LOCAL_RANK", "-1")) >= 0:
                        dist.all_reduce(o_proj.head_norm_r, op=dist.ReduceOp.SUM)
                        dist.all_reduce(o_proj.attn_norm_r, op=dist.ReduceOp.SUM)
                        o_proj.num_tokens = torch.tensor(o_proj.num_tokens, device=o_proj.head_norm_r.device)
                        dist.all_reduce(o_proj.num_tokens, op=dist.ReduceOp.SUM)
                    state_dict[f'{i}.{name}.head_norm_r'] = (o_proj.head_norm_r / o_proj.num_tokens).cpu()
                    state_dict[f'{i}.{name}.attn_norm_r'] = (o_proj.attn_norm_r / o_proj.num_tokens).cpu()
                    state_dict[f'{i}.{name}.sparse_head_ratio'] = (o_proj.sparse_head_cnt / o_proj.num_tokens).cpu()
                    state_dict[f'total_tokens'] = o_proj.num_tokens
                    o_proj.enable_hook = False
                except Exception as e:
                    pass
            for name in fc2_name:
                try:
                    fc2 = transformer.get_submodule(name)
                    if int(os.getenv("LOCAL_RANK", "-1")) >= 0:
                        dist.all_reduce(fc2.act_value, op=dist.ReduceOp.SUM)
                        dist.all_reduce(fc2.act_num, op=dist.ReduceOp.SUM)
                        fc2.num_tokens = torch.tensor(fc2.num_tokens, device=fc2.act_value.device)
                        dist.all_reduce(fc2.num_tokens, op=dist.ReduceOp.SUM)
                    state_dict[f'{i}.{name}.avg_act_value'] = (fc2.act_value / fc2.num_tokens).cpu()
                    state_dict[f'{i}.{name}.act_ratio'] = (fc2.act_num / fc2.num_tokens).cpu()
                    state_dict[f'{i}.{name}.num_tokens'] = fc2.num_tokens
                    fc2.enable_hook = False
                except Exception as e:
                    print(e)
            transformer.enable_hook = False
    else:
        for module_name, fc2 in transformers.named_modules():
            if isinstance(fc2, nn.Linear) and module_name.split('.')[-1] in transformers.fc2_name:
                if int(os.getenv("LOCAL_RANK", "-1")) >= 0:
                    dist.all_reduce(fc2.act_value, op=dist.ReduceOp.SUM)
                    dist.all_reduce(fc2.act_num, op=dist.ReduceOp.SUM)
                    fc2.num_tokens = torch.tensor(fc2.num_tokens, device=fc2.head_norm_r.device)
                    dist.all_reduce(fc2.num_tokens, op=dist.ReduceOp.SUM)
                state_dict[f'{module_name}.avg_act_value'] = (fc2.act_value / fc2.num_tokens).cpu()
                state_dict[f'{module_name}.act_ratio'] = (fc2.act_num / fc2.num_tokens).cpu()
                state_dict[f'{module_name}.num_tokens'] = fc2.num_tokens
                fc2.enable_hook = False
    return state_dict

