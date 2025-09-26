import copy
import os
import random
from functools import partial
from typing import Sequence

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_forecast import Exp_Forecast
from layers.prune_mask import MaskedLayer, add_masks_, Mask

from utils.tools import EarlyStopping


class Exp_Score(Exp_Forecast):
    def __init__(self, args):
        args.freeze = True
        self.iterative_epochs = 1
        self.prune_ratio = args.prune_ratio
        self.pruner_type = args.pruner_type
        self.sparse_model = args.model in ['TimeMoE']
        super().__init__(args)

    def _build_model(self, model=None, framework_class=None):
        wrapper = partial(add_masks_, track_grad=self.pruner_type,
                          valid_name_fn=None if self.args.model != 'TimeMoE' else lambda x: not x.endswith('gate'))
        model = Exp_Forecast._build_model(self, model, wrapper)
        return model

    def train(self, setting, datas=None):
        while True:
            try:
                train_data, train_loader = self._get_data(flag='train')
                # vali_data, vali_loader = self._get_data(flag='val')
                imps = self.prune(train_data, train_loader)
                return self.model
            except torch.cuda.OutOfMemoryError:
                print(
                    f"OutOfMemoryError at batch_size {self.args.batch_size}, reducing to {self.args.batch_size//2}"
                )
                self.args.batch_size //= 2

    def test(self, setting):
        if not self.args.do_training:
            test_data, test_loader = data_provider(self.args, flag='test', label_len=self.args.label_len,
                                                   pred_len=self.args.patch_len if self.args.autoregressive else None)
            imps = self.prune(test_data, test_loader, None, None)


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
                return score, False
        return score, False

    def prune(self, train_data, train_loader, valid_data=None, valid_loader=None):
        result_path = os.path.join(f'results/{self.args.model}_{self.args.model_size}_'
                                   f'{self.args.data_path.split(".")[0]}{self.args.subset_ratio}_'
                                   f'{self.args.seq_len}{"" if self.args.autoregressive else ("_" + str(self.args.pred_len))}'
                                   f'_{"train" if self.args.do_training else "test"}_score.npz')
        if not os.path.exists(os.path.dirname(result_path)):
            os.makedirs(result_path)

        if self.sparse_model:
            self._model.apply_aux_loss = False
            self._model.register_batch_id_handler_()
        self.model.zero_grad()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        if valid_loader is not None:
            mse = super().vali(valid_data, valid_loader, criterion=self._select_criterion())
            print('Valid mse: {:.4f}'.format(mse))
            early_stopping(mse, self.model, None)
            torch.cuda.empty_cache()
        else:
            early_stopping(1e8, self.model, None)
        # self.model.training = True
        criterion = nn.MSELoss(reduction='sum')
        target_module_names = list()
        for name, module in self.model.named_modules():
            if isinstance(module, MaskedLayer):
                target_module_names.append(name)

        imps = {}
        for name in target_module_names:
            if self.args.mask_in:
                imps[name + '.mask_in'] = torch.zeros_like(self.model.get_submodule(name + '.mask_in').mask)
            if self.args.mask_out:
                imps[name + '.mask_out'] = torch.zeros_like(self.model.get_submodule(name + '.mask_out').mask)
        for batch in tqdm(train_loader, miniters=1, mininterval=10, leave=False):
            self.model.training = True
            batch_x, batch_y = batch[:2]
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            if self.args.autoregressive or getattr(self._model, 'enable_loss_fn', False):
                outputs, loss = self.model(batch_x, labels=batch_y)
            else:
                outputs = self.model(batch_x)[:, :self.args.pred_len, :]
                loss = criterion(outputs, batch_y[:, :outputs.size(1)])
            loss.backward()

        with torch.no_grad():
            for name in target_module_names:
                module = self.model.get_submodule(name)
                for mask_name in ['mask_in', 'mask_out']:
                    if getattr(self.args, mask_name) and getattr(module, mask_name).mask.grad is not None:
                        score = self._score(getattr(module, mask_name).mask)
                        score, unused = self._handle_sparse_model(getattr(module, mask_name), score)
                        k = name + '.' + mask_name
                        if not unused:
                            imps[k] += score
        self.model.zero_grad()
        imps = {k: v for k, v in imps.items() if isinstance(v, torch.Tensor)}
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            np.savez(result_path, **{k: v.float().cpu().numpy() for k, v in imps.items()})
        if self.sparse_model:
            self._model.remove_batch_id_handler_()
        return imps