import os
from heapq import nlargest

import torch
from torch import nn
from tqdm import tqdm

from exp.exp_forecast import Exp_Forecast
from layers.prune_mask import MaskedLayer, add_masks_, Mask
from utils.monitor import wrap_model, collect_states, get_module_name


class Exp_Statistics(Exp_Forecast):
    def __init__(self, args):
        args.freeze = True
        self.prune_ratio = args.prune_ratio
        self.pruner_type = args.pruner_type
        self.prune_ema = args.prune_ema
        self.pruned = False
        self.sparse_model = args.model in ['TimeMoE']
        self.prune_expert_threshold = args.prune_expert_threshold
        self.iterative_epochs = 1
        self.tunable = {}
        super().__init__(args)

    def _build_model(self, model=None, framework_class=None):
        model = Exp_Forecast._build_model(self, model, wrap_model)
        return model

    def _prune(self, state_dict):
        all_head_norm_r = []
        all_act_value = []
        all_act_ratio = []
        for k, v in state_dict.items():
            if k.endswith('head_norm_r'):
                all_head_norm_r.append(v)
            if k.endswith('act_value'):
                all_act_value.append(v)
            if k.endswith('act_ratio'):
                all_act_ratio.append(v)
        # if all_head_norm_r:
        #     all_head_norm_r = torch.concat(all_head_norm_r)
        # all_act_value = torch.concat(all_act_value)
        # all_act_ratio = torch.concat(all_act_ratio)

        kwargs = get_module_name(self.args.model, self._model)
        o_proj_name = kwargs["o_proj_name"]
        fc2_name = kwargs["fc2_name"]
        transformers = getattr(self._model, "transformers", self._model)
        if getattr(self.args, 'prune_head', False):
            # prune_num = int(len(all_head_norm_r) * self.args.prune_head)
            # threshold = torch.topk(all_head_norm_r, k=prune_num, largest=False)[0][-1].item()
            threshold = self.args.prune_head
            if isinstance(transformers, (nn.ModuleList, list)):
                for i, transformer in enumerate(transformers):
                    for name in o_proj_name:
                        try:
                            o_proj = transformer.get_submodule(name)
                        except Exception as e:
                            continue
                        mask = state_dict[f'{i}.{name}.head_norm_r'].to(self.device) <= threshold
                        if mask.any():
                            mask = mask.unsqueeze(-1).repeat(1, len(o_proj.mask_in.mask) // len(mask)).flatten()
                            o_proj.mask_in.mask.data.masked_fill_(mask, 0)

        # prune_num = max(1, int(len(all_act_value) * self.args.prune_ratio))
        # threshold = torch.topk(all_act_value, k=prune_num, largest=False)[0][-1].item()
        threshold = self.args.act_prob_threshold
        if isinstance(transformers, (nn.ModuleList, list)):
            for i, transformer in enumerate(transformers):
                for name in fc2_name:
                    try:
                        fc2 = transformer.get_submodule(name)
                    except Exception as e:
                        continue
                    mask = state_dict[f'{i}.{name}.act_ratio'].to(self.device) <= threshold
                    if mask.any():
                        fc2.mask_in.mask.data.masked_fill_(mask, 0)
        else:
            transformers.fc2_name = fc2_name
            for module_name, fc2 in transformers.named_modules():
                if isinstance(fc2, nn.Linear) and module_name.split('.')[-1] in fc2_name:
                    mask = state_dict[f'{module_name}.act_ratio'].to(self.device) <= threshold
                    if mask.any():
                        fc2.mask_in.mask.data.masked_fill_(mask, 0)

        if self.sparse_model:
            all_experts, all_ffns = self._model.experts, self._model.ffns
            expert_use_frequency = [
                [
                    state_dict[f'{i}.ffn_layer.experts.{j}.down_proj.num_tokens'] / state_dict['total_tokens']
                    for j in range(len(all_experts[0]))
                ]
                for i in range(len(all_experts))
            ]
            for i, (experts, ffn) in enumerate(zip(all_experts, all_ffns)):
                pruned = []
                if self.prune_expert_threshold < 0:
                    selected = nlargest(self._model.num_experts_per_tok, enumerate(expert_use_frequency[i]), key=lambda x: x[1])
                for j, expert in enumerate(experts):
                    if (self.prune_expert_threshold >= 0 and expert_use_frequency[i][j] <= self.prune_expert_threshold
                        or self.prune_expert_threshold < 0 and j not in selected):
                        for module in expert.modules():
                            if isinstance(module, MaskedLayer):
                                module.mask_in.mask.data.zero_()
                                module.mask_out.mask.data.zero_()
                        pruned.append(j)
                        experts[j] = nn.Identity()
                if len(pruned):
                    ffn.register_buffer('pruned_expert_ids',
                                        torch.tensor(pruned, dtype=torch.int64, device=self.device),
                                        persistent=True)

    @torch.no_grad()
    def merge_weights(self, setting=None):
        if self.pruned:
            return
        self.model.eval()
        pred_len = 128 if self.args.model == 'TimesFM' else self.args.pred_len
        if self.args.model == 'Chronos':
            pred_len = 64
        result_path = os.path.join(f'results/{self.args.model}_{self.args.model_size}_'
                                   f'{self.args.data_path.split(".")[0]}{self.args.subset_ratio}_'
                                   f'{self.args.seq_len if self.args.model != "TimeMoE" else "4096"}_'
                                   f'{"" if self.args.autoregressive else (str(pred_len) + "_")}'
                                   # f'{"finetuned_" if self.args.reload else ""}'
                                   f'{"train" if self.args.do_training else "test"}_stats.pt')
        if os.path.exists(result_path):
            print("Loading", result_path)
            state_dict = torch.load(result_path, map_location=self.device)
        else:
            if self.args.do_training:
                train_data, data_loader = self._get_data(flag='train')
                valid_data, valid_loader = self._get_data(flag='train')
            else:
                test_data, data_loader = self._get_data(flag='test', enlarge_bs=4)
            for i, batch in enumerate(tqdm(data_loader, miniters=1, mininterval=10, leave=False)):
                batch = [d.to(self.device) for d in batch]
                self.model(batch[0], *batch[2:])
            for i, batch in enumerate(tqdm(valid_loader, miniters=1, mininterval=10, leave=False)):
                batch = [d.to(self.device) for d in batch]
                self.model(batch[0], *batch[2:])
            state_dict = collect_states(getattr(self._model, "transformers", self._model))
            if int(os.getenv("LOCAL_RANK", "0")) >= 0:
                torch.save(state_dict, result_path)

        self.model = add_masks_(self.model, track_grad=False)
        self._prune(state_dict)

        self.pruned = True
        self.model.requires_grad_(True)

        sparsity = {}
        if hasattr(self._model, 'transformers'):
            for i, layer in enumerate(self._model.transformers):
                for k, module in layer.named_modules():
                    if 'Chronos' in self.args.model:
                        k = ('encoder.' if i < len(self._model.transformers) // 2 else 'decoder.') + k
                    if isinstance(module, MaskedLayer):
                        if k not in sparsity:
                            if 'Chronos' in self.args.model:
                                sparsity[k] = torch.ones(len(self._model.transformers) // 2)
                            else:
                                sparsity[k] = torch.ones(len(self._model.transformers))
                        in_dim = module.mask_in.mask.sum().item()
                        out_dim = module.mask_out.mask.sum().item()
                        j = i % (len(self._model.transformers) // 2) if 'Chronos' in self.args.model else i
                        sparsity[k][j] = 1 - (in_dim * out_dim) / (module.weight.shape[0] * module.weight.shape[1])
            print(sparsity)

        self._model.merge_weights_()
        torch.cuda.empty_cache()
        model_params = sum([(param != 0).sum() for param in self.model.parameters()])
        print('Number of Params: {:.2f} M'.format(model_params / 1e6))

        if self.sparse_model:
            self._model.register_batch_id_handler_()
            self._model.remove_batch_id_handler_()
        return

    def load_state_dict(self, *args, **kwargs):
        if self.sparse_model:
            for ffn in self._model.ffns:
                ffn.pruned_expert_ids = None
        return super().load_state_dict(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if getattr(self, 'tunable', None):
            state_dict['tunable'] = self.tunable
        return state_dict
