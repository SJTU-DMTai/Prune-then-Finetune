import collections
import os
import typing
import warnings
from functools import partial
from heapq import nlargest

import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.distributed as dist

from exp.exp_forecast import Exp_Forecast
from layers.prune_mask import MaskedLayer, add_masks_, Mask
from utils.dataset import get_train_val_data_from_gluonts


class Exp_Prune(Exp_Forecast):
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
        valid_name_fn = None
        if self.args.model == 'TimeMoE':
            valid_name_fn = lambda x: not x.endswith('gate')
        elif self.args.model == 'TTM':
            if self.args.probing:
                valid_name_fn = lambda x: not x.endswith('base_forecast_block')
            else:
                valid_name_fn = lambda x: x.endswith('base_forecast_block')
        wrapper = partial(add_masks_, track_grad=self.pruner_type,
                          valid_name_fn=valid_name_fn, )
        model = Exp_Forecast._build_model(self, model, wrapper)
        if self.args.probing and self.args.model == 'TTM':
            model.model.head.base_forecast_block.mask_out.requires_grad = False
        return model

    def _select_optimizer(self):
        if not self.pruned:
            self.model_optim = None
            return None
        else:
            return super()._select_optimizer()

    @torch.no_grad()
    def prune_expert(self):
        result_path = os.path.join(f'results/{self.args.model}_{self.args.model_size}_'
                                   f'{self.args.data_path.split(".")[0]}{self.args.subset_ratio}_'
                                   f'{self.args.seq_len}_'
                                   f'expert_frequency.npy')
        if not os.path.exists('results'):
            os.makedirs('results')

        all_experts, all_ffns = self._model.experts, self._model.ffns
        for name, module in all_experts[0][0].named_modules():
            if isinstance(module, MaskedLayer):
                break
        if os.path.exists(result_path):
            print("Loading", result_path)
            expert_use_frequency = np.load(result_path)
        else:
            _, vali_loader = self._get_data(flag='train', enlarge_bs=4)
            for i, batch in enumerate(tqdm(vali_loader, miniters=1, mininterval=10, leave=False)):
                batch = [d.to(self.device) for d in batch]
                self.model(batch[0], *batch[2:])
            _, vali_loader = self._get_data(flag='val', enlarge_bs=4)
            for i, batch in enumerate(tqdm(vali_loader, miniters=1, mininterval=10, leave=False)):
                batch = [d.to(self.device) for d in batch]
                self.model(batch[0], *batch[2:])
            if self.args.use_multi_gpu:
                dist.barrier()
            token_cnts = [[expert.get_submodule(name).mask_out.read_use_count()[1] for expert in experts] for experts in all_experts]
            token_cnts = np.array(token_cnts)
            print('rank', os.getenv('LOCAL_RANK', 0), ':', token_cnts)
            num_token = sum(token_cnts[0]) // self._model.num_experts_per_tok
            expert_use_frequency = token_cnts / num_token
            np.save(result_path, expert_use_frequency)
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
                                    torch.tensor(pruned, dtype=torch.int64,
                                                    device=expert.get_submodule(name).mask_in.mask.device),
                                    persistent=True)

    def prune(self, setting):
        if self.sparse_model:
            self._model.apply_aux_loss = False
            self._model.register_batch_id_handler_()
            batch_size = self.args.batch_size
            self.args.batch_size *= 8
            while self.args.batch_size:
                try:
                    self.prune_expert()
                    break
                except torch.cuda.OutOfMemoryError:
                    self.args.batch_size //= 2
                    print("Reduce batch size to", self.args.batch_size)
                    torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            self.args.batch_size = batch_size
            if self.args.only_prune_expert:
                self.merge_weights()
                return
            if self.args.use_multi_gpu:
                dist.barrier()

        self.target_module_names = list()
        self.imps = {}
        for name, module in self.model.named_modules():
            if isinstance(module, MaskedLayer):
                self.target_module_names.append(name)
        batch_size = self.args.batch_size * int(os.getenv('WORLD_SIZE', 1))
        train_data, vali_data = None, None
        if self.args.gift_eval:
            train_data, vali_data, self.scaler = get_train_val_data_from_gluonts(self.args)
        train_data, train_loader = self._get_data(flag='train', data_set=train_data)
        vali_data, vali_loader = self._get_data(flag='val', data_set=vali_data)
        if self.args.num_batch_per_epoch <= 0:
            if self.args.macro_batch_size > batch_size:
                self.args.grad_accumulation = min(self.args.macro_batch_size // batch_size, len(train_loader))
            self.prune_ratio = self.args.prune_ratio_per_epoch / len(train_loader) * self.args.grad_accumulation
        else:
            if self.args.macro_batch_size > batch_size:
                self.args.grad_accumulation = min(self.args.macro_batch_size // batch_size, self.args.num_batch_per_epoch)
            self.prune_ratio = self.args.prune_ratio_per_epoch / self.args.num_batch_per_epoch * self.args.grad_accumulation

        print('Prune ratio per batch:', self.prune_ratio)

        val_metric = super().train(setting, (train_data, train_loader, vali_data, vali_loader))
        self.merge_weights()
        if self.sparse_model:
            self._model.remove_batch_id_handler_()
        return val_metric

    def pre_train_epoch(self,):
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train(self.pruned and self.args.enforce_dropout)

    def post_train_epoch(self, early_stopping):
        if not self.pruned and self.args.max_prune_ratio < 1:
            self.model.eval()
            model_size = sum([param.numel() for param in self.model.parameters()])
            self.model.eval()
            pruned_num2 = sum([(param.abs() == 0).sum() for param in self.model.parameters()])
            if self.args.max_prune_ratio < 1:
                if pruned_num2 / model_size + self.prune_ratio > self.args.max_prune_ratio:
                    early_stopping.early_stop = True


    def mark_tunable_module(self):
        self.tunable = {}
        for k, v in self.imps.items():
            name = '.'.join(k.split('.')[:-1])
            v = v.sum().item()
            if v < 0:
                self.tunable[name] = min(v, self.tunable[name]) if name in self.tunable else v
        return self.tunable

    def _score(self, mask):
        score = - mask.grad * mask.data
        if self.args.ema_abs:
            score = score.abs()
        return score

    def _handle_sparse_model(self, mask: Mask, score):
        if self.sparse_model and hasattr(mask, 'batch_id'):
            sample_cnt, token_cnt = mask.read_use_count(self.args.use_multi_gpu)
            if sample_cnt == 0:
                return torch.zeros_like(score), True
            else:
                if self.args.average_expert_score:
                    return score * (self.args.batch_size * int(os.getenv("WORLD_SIZE", "1")) / sample_cnt), False
                else:
                    return score, False
        return score, False

    def train_step(self, loss, model_optim, iter_count):
        if self.pruned:
            return super().train_step(loss, model_optim, iter_count)
        loss.backward()
        if iter_count % self.args.grad_accumulation == 0:
            with torch.no_grad():
                unused_names = set()
                for name in self.target_module_names:
                    module = self._model.get_submodule(name)
                    for mask_name in ['mask_in', 'mask_out']:
                        if getattr(self.args, mask_name):
                            if getattr(module, mask_name).mask.grad is None:
                                unused_names.add(name + '.' + mask_name)
                            else:
                                score = self._score(getattr(module, mask_name).mask)
                                score, unused = self._handle_sparse_model(getattr(module, mask_name), score)
                                k = name + '.' + mask_name
                                if unused:
                                    unused_names.add(k)
                                else:
                                    if k not in self.imps:
                                        self.imps[k] = score
                                    else:
                                        self.imps[k] = self.prune_ema * score + self.imps[k] * (1 - self.prune_ema)
                self.model.zero_grad()
                self._prune(self.imps, unused_names)

    @torch.no_grad()
    def _prune(self, imps, unused_names):
        scores = torch.cat([v for k, v in imps.items() if k not in unused_names]).abs()
        scores = scores[scores > 0]
        prune_num = math.ceil(len(scores) * self.prune_ratio)
        if len(scores) > prune_num:
            threshold = torch.topk(scores, prune_num, largest=False)[0][-1]
        else:
            threshold = 0
        if self.args.use_multi_gpu:
            dist.barrier()
            dist.all_reduce(threshold, op=dist.ReduceOp.MIN)
        keys = []
        if self.args.mask_in:
            keys.append('.mask_in')
        if self.args.mask_out:
            keys.append('.mask_out')
        for k, imp in imps.items():
            if k in unused_names:
                continue
            module = self._model.get_submodule('.'.join(k.split('.')[:-1]))
            allow = (imp.abs() <= threshold) | (imp == 0)
            if allow.any():
                if k.endswith('.mask_out'):
                    module.mask_out.mask.data[allow] = 0
                else:
                    module.mask_in.mask.data[allow] = 0
                imps[k][allow] = 0

    def merge_weights(self, setting=None):
        self.pruned = True
        if not self.args.lora_rank:
            self.model.requires_grad_(True)

        sparsity = {}
        try:
            for i, layer in enumerate(self._model.transformers): # Raise NotImplementedError
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
        except NotImplementedError:
            warnings.warn(f"{type(self._model)} has not defined its transformers.")

        self._model.merge_weights_()
        torch.cuda.empty_cache()
        model_params = sum([(param != 0).sum() for param in self.model.parameters()])
        print('Number of Params: {:.2f} M'.format(model_params / 1e6))
        model_params = sum([(param != 0).sum() for param in self.model.parameters() if param.requires_grad])
        print('Number of Tunable Params: {:.2f} M'.format(model_params / 1e6))

        if self.sparse_model:
            self._model.register_batch_id_handler_()
            self._model.remove_batch_id_handler_()
        return

    @torch.no_grad()
    def analysis(self):
        import seaborn as sns
        import numpy as np
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        if hasattr(self._model, 'transformer_names'):
            sparsity = {'all': {}, 'in': {}, 'out': {}}
            n_layer = len(self._model.transformers) // 2 if 'Chronos' in self.args.model else (
                len(self._model.transformers))
            for i, layer in enumerate(self._model.transformers):
                for k, module in layer.named_modules():
                    if 'Chronos' in self.args.model:
                        k = ('encoder.' if i < len(self._model.transformers) // 2 else 'decoder.') + k
                    if isinstance(module, MaskedLayer):
                        if k not in sparsity['all']:
                            sparsity['all'][k] = torch.ones(n_layer) * module.in_features * module.out_features
                            sparsity['in'][k] = torch.ones(n_layer) * module.in_features
                            sparsity['out'][k] = torch.ones(n_layer) * module.out_features
        self.merge_weights()
        if hasattr(self._model, 'transformer_names'):
            n_layer = len(self._model.transformers) // 2 if 'Chronos' in self.args.model else (
                len(self._model.transformers))
            for i, layer in enumerate(self._model.transformers):
                for k, module in layer.named_modules():
                    if 'Chronos' in self.args.model:
                        k = ('encoder.' if i < len(self._model.transformers) // 2 else 'decoder.') + k
                    if isinstance(module, MaskedLayer):
                        in_dim = module.weight.shape[1] if module.mask_in is None else module.mask_in.mask.sum().item()
                        out_dim = module.weight.shape[0] if module.mask_out is None else module.mask_out.mask.sum().item()
                        j = i % (len(self._model.transformers) // 2) if 'Chronos' in self.args.model else i
                        sparsity['all'][k][j] = 1 - (in_dim * out_dim) / sparsity['all'][k][j]
                        sparsity['in'][k][j] = 1 - in_dim / sparsity['in'][k][j]
                        sparsity['out'][k][j] = 1 - out_dim / sparsity['out'][k][j]
            for k in sparsity['all']:
                sparsity['all'][k].clamp_(min=0, max=1)
                sparsity['in'][k].clamp_(min=0, max=1)
                sparsity['out'][k].clamp_(min=0, max=1)
            if self.sparse_model:
                for i, ffn in enumerate(self._model.ffns):
                    if ffn.pruned_expert_ids is not None:
                        for j in ffn.pruned_expert_ids:
                            for k in sparsity:
                                for name in sparsity[k]:
                                    if f'experts.{j}' in name:
                                        sparsity[k][name][i] = 1
            merged_sparsity = {'all': {}, 'in': {}, 'out': {}}
            translation = {plot_name: name for plot_name, name
                         in zip([r'${\bf W}^Q$', r'${\bf W}^K$', r'${\bf W}^V$', r'${\bf W}^O$',
                                 r'${\bf W}^{\tt{gate}}$', r'${\bf W}^{\tt{up}}$', r'${\bf W}^{\tt{down}}$'],
                                self._model.transformer_names,)
                         if name is not None and 'gate' not in name}
            for _type in sparsity.keys():
                for plot_name, k in translation.items():
                    vs = {}
                    for name, v in sparsity[_type].items():
                        if name.endswith(k) and not ('Chronos' in self.args.model and
                                                     'Self' in name and 'decoder' in name and
                                                     ('q' in name or 'k' in name)):
                            vs[name] = v
                    if vs:
                        if 'Chronos' in self.args.model:
                            merged_sparsity[_type][plot_name] = torch.cat(
                                [torch.stack([v for name, v in vs.items() if 'encoder' in name]).mean(0),
                                 torch.stack([v for name, v in vs.items() if 'decoder' in name]).mean(0)], -1)
                        else:
                            merged_sparsity[_type][plot_name] = torch.stack(list(vs.values())).mean(0)
            print(merged_sparsity)
            if 'Chronos' in self.args.model:
                n_layer *= 2
            for _type, spr in merged_sparsity.items():
                y_ticks = list(spr.keys())
                fig = plt.figure(figsize=(0.42 * n_layer, 0.42 * len(y_ticks)), dpi=150)
                ax = sns.heatmap(torch.stack(list(spr.values())), annot=True, cmap="Blues", fmt=".0%",
                            xticklabels=np.arange(n_layer),
                            yticklabels=y_ticks, vmin=0, vmax=1, cbar=False)
                ax.set_xlabel('Layer ID')
                plt.savefig(f'figs/{self.args.model}_{self.args.model_size}_{self.args.data_path.split(".")[0]}_'
                            f'{self.args.pred_len}_{self.args.prune_ratio_per_epoch}_{_type}.pdf',
                            bbox_inches='tight', pad_inches=0.05)
        super().analysis()

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
