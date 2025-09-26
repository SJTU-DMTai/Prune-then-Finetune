import copy
import importlib
import json
import logging
import os
import time
import warnings

import math
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DistributedSampler
from tqdm import tqdm
from transformers import AdamW

from data_provider import data_factory
from data_provider.data_factory import data_provider, get_dataloader
from exp.exp_basic import Exp_Basic
from utils.dataset import get_train_val_data_from_gluonts, get_test_data_from_gluonts
from utils.metrics import update_metrics, calculate_metrics, MAPE
from utils.tools import EarlyStopping, LargeScheduler, test_params_flop

warnings.filterwarnings('ignore')


class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        self.args = args
        super().__init__(args)
        self.scaler = None

    def _build_model(self, model=None, framework_class=None):
        self.args.device = self.device
        if model is None:
            model = importlib.import_module(f'models.{self.args.model}').Model(self.args)
        if self.args.freeze:
            model.requires_grad_(False)
        elif self.args.probing:
            model.linear_probing_()
        model_params = sum([param.nelement() for param in model.parameters()])
        if framework_class is not None:
            if isinstance(framework_class, list):
                for cls in framework_class:
                    model = cls(model, **vars(self.args))
            else:
                model = framework_class(model, **vars(self.args))
            new_model_params = sum([param.nelement() for param in model.parameters()])
            print(f'Number of Params: {model_params} -> {new_model_params} (+{new_model_params - model_params})')
        trainable_params = sum([param.nelement() if param.requires_grad else 0 for param in model.parameters()])
        print(f'Trainable Params: {trainable_params}', '({:.1f}%)'.format(trainable_params / model_params * 100))
        if getattr(self.args, 'compile', None):
            model = torch.compile(model)
        return model

    def _build_ddp_model(self):
        if not self.ddp_model and self.args.use_multi_gpu and self.args.use_gpu:
            self.model = DDP(self._model.cuda(), device_ids=[self.args.local_rank],
                        find_unused_parameters=self.args.find_unused_parameters)
            self.ddp_model = True

    def _get_data(self, flag, enlarge_bs=1, data_set=None, **kwargs):
        if flag == 'val':
            enlarge_bs = 4
        if enlarge_bs > 1:
            bs, self.args.batch_size = self.args.batch_size, self.args.batch_size * enlarge_bs
        if data_set is None:
            data_set, data_loader = data_provider(self.args, flag, **kwargs)
        else:
            data_loader = get_dataloader(data_set, self.args, flag, **kwargs)
        if enlarge_bs > 1:
            self.args.batch_size = bs
        if flag == 'train' and 0 < self.args.num_batch_per_epoch < 1:
            self.args.num_batch_per_epoch = int(self.args.num_batch_per_epoch * len(data_set) / self.args.batch_size /
                                                (dist.get_world_size() if self.args.use_multi_gpu else 1))
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.use_weight_decay and self.args.weight_decay:
            from transformers.trainer_pt_utils import get_parameter_names
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            model_optim = optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, betas=(0.9, 0.95))
        else:
            model_optim = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay if self.args.use_weight_decay else 0)
        self.model_optim = model_optim
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'mse':
            criterion = nn.MSELoss()
        elif self.args.loss == "mae":
            criterion = nn.L1Loss()
        else:
            raise NotImplementedError
        return criterion

    @torch.no_grad()
    def vali(self, vali_data=None, vali_loader=None, criterion=None, epoch=0, flag='vali'):
        if vali_data is None:
            vali_data, vali_loader = self._get_data(flag='val')
        if self.args.val_loss:
            if self.args.val_loss.lower() == 'mape':
                criterion = MAPE
            elif self.args.val_loss.lower() == 'mae':
                criterion = nn.L1Loss()
        elif criterion is None:
            criterion = self._select_criterion()
        total_loss = []
        total_count = []
        self.model.eval()
        vali_loader = tqdm(vali_loader, miniters=1, mininterval=10, leave=False)
        for i, batch in enumerate(vali_loader):
            batch = [d.to(self.device) for d in batch]
            batch_x, batch_y = batch[:2]
            if self.args.valid_autoregressive:
                outputs, loss = self.model(batch_x, *batch[2:], labels=batch_y)
            else:
                outputs = self.model(batch_x, *batch[2:])
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                outputs = outputs[:, :self.args.pred_len, :]
                loss = criterion(outputs, batch_y)

            total_loss.append(loss.item())
            total_count.append(batch_x.shape[0])

        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
        self.model.train()
        return total_loss

    def pre_train_epoch(self,):
        if self.args.disable_dropout:
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.train(False)

    def post_train_epoch(self, early_stopping):
        pass

    def train(self, setting, datas=None):
        trainable_params = sum([param.nelement() if param.requires_grad else 0 for param in self.model.parameters()])
        print(f'Trainable Params: {trainable_params}')
        self._build_ddp_model()
        if datas is None:
            train_data, vali_data = None, None
            if self.args.gift_eval:
                train_data, vali_data, self.scaler = get_train_val_data_from_gluonts(self.args)
                if self.args.model == 'TTM':
                    if self.args.target_dim < 10:
                        self.args.batch_size = 64
                    else:
                        self.args.batch_size = 16
                    if len(train_data) <= 1_000:
                        self.args.batch_size = 8
                    elif len(train_data) > 100_000:
                        self.args.batch_size = 512
                print("Batch size:", self.args.batch_size)
            train_data, train_loader = self._get_data(flag='train', data_set=train_data)
            vali_data, vali_loader = self._get_data(flag='val', data_set=vali_data)
        else:
            train_data, train_loader, vali_data, vali_loader = datas

        batch_size = self.args.batch_size * int(os.getenv('WORLD_SIZE', 1))
        if self.args.macro_batch_size > batch_size:
            self.args.grad_accumulation = min(self.args.macro_batch_size // batch_size, len(train_loader))
            if self.args.num_batch_per_epoch > 1:
                self.args.grad_accumulation = min(self.args.grad_accumulation, self.args.num_batch_per_epoch)
        print("Gradient Accumulation: {}".format(self.args.grad_accumulation))

        self.max_steps = min(2, len(train_loader))

        if self.args.checkpoints:
            path = os.path.join(self.args.checkpoints, setting)
            if not os.path.exists(self.args.checkpoints) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
                os.makedirs(self.args.checkpoints)
        else:
            path = None

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if model_optim is not None:
            if self.args.cosine:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax,
                                                                       eta_min=self.args.learning_rate / 1000)
            elif self.args.num_batch_per_epoch:
                scheduler = None
            elif self.args.gift_eval and self.args.finetune_epochs > 1:
                scheduler = OneCycleLR(model_optim, self.args.learning_rate, epochs=self.args.finetune_epochs,
                                       steps_per_epoch=math.ceil(len(train_data) / self.args.batch_size),)
            else:
                scheduler = LargeScheduler(self.args, model_optim)
        else:
            scheduler = None

        def check_epoch(epoch_id, train_loss=None):
            if self.args.finetune_epochs == 1 and self.args.model == 'TimeMoE':
                return
            torch.cuda.empty_cache()
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if train_loss:
                print("Epoch: {} | Train Loss: {:.4f} | Vali Loss: {:.4f}".format(epoch_id, train_loss, vali_loss))
            else:
                print("Epoch: {} | Vali Loss: {:.4f}".format(epoch_id, vali_loss))
            if scheduler is not None:
                if isinstance(scheduler, LargeScheduler):
                    scheduler.schedule_epoch(epoch_id)
                elif self.args.cosine:
                    scheduler.step()
            early_stopping(vali_loss, self, path)
            torch.cuda.empty_cache()

        iter_count = 0
        subepoch_id = 0
        num_sample_per_batch = train_loader.batch_size * self.args.grad_accumulation
        num_full_batch = (len(train_loader) // self.args.grad_accumulation) * self.args.grad_accumulation
        for epoch in range(self.args.finetune_epochs):
            loss_train = torch.tensor(0., device=self.device)
            count = torch.tensor(0., device=self.device)

            self.model.train()
            epoch_time = time.time()

            self.pre_train_epoch()
            for i, batch in enumerate(tqdm(train_loader, miniters=1, mininterval=10, leave=False)):
                batch = [d.to(self.device) for d in batch] # No timestamps
                batch_x, batch_y = batch[:2]
                iter_count += 1

                if self.args.autoregressive or getattr(self._model, "enable_loss_fn", False):
                    outputs, loss = self.model(batch_x, *batch[2:], labels=batch_y)
                else:
                    # only use the forecast window to calculate loss
                    outputs = self.model(batch_x, *batch[2:])
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = criterion(outputs[:, :self.args.pred_len, :], batch_y)
                if torch.isnan(loss):
                    raise Exception('Loss is nan at count'.format(count.item()),
                                    'Previous loss:', loss_train.item() / count.item())
                loss_train += loss.item()

                if self.args.grad_accumulation > 1:
                    if self.args.finetune_epochs == 1 and i >= num_full_batch: # last batch
                        num_sample_per_batch = len(batch_x) + (len(train_loader) - num_full_batch - 1) * train_loader.batch_size
                        if i == len(train_loader) - 1:
                            self.args.grad_accumulation = 1 # enforce backward soon
                    loss = loss * (len(batch_x) / num_sample_per_batch)
                count += 1
                subepoch_id += 1

                self.train_step(loss, model_optim, iter_count)
                if isinstance(scheduler, OneCycleLR):
                    scheduler.step()

                if (self.args.num_batch_per_epoch and subepoch_id >= self.args.valid_begin and
                        subepoch_id % int(self.args.num_batch_per_epoch) == 0):
                    check_epoch(subepoch_id)
                    if early_stopping.early_stop:
                        break

            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(loss_train, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_train.item() / count.item()
            if not early_stopping.early_stop:
                self.post_train_epoch(early_stopping)
                if epoch >= self.args.valid_begin and self.args.num_batch_per_epoch <= 0:
                    check_epoch(subepoch_id if self.args.num_batch_per_epoch else epoch + 1, train_loss)
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # elif early_stopping.counter == 1:
            #     if path:
            #         if self.args.use_multi_gpu:
            #             dist.barrier()
            #         best_model_path = path + '_checkpoint.pth'
            #         print('Save checkpoint to', best_model_path)
            #         torch.save(early_stopping.best_checkpoint, best_model_path)
            if path and self.args.save_per_epoch:
                if self.args.use_multi_gpu:
                    dist.barrier()
                torch.save(self.state_dict() if self.flag_load_state_dict else self._model, path + '_checkpoint.pth' + str(subepoch_id))

            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch + 1)

        if not (self.args.finetune_epochs == 1 and self.args.model == 'TimeMoE'):
            print('Best Valid MSE:', -early_stopping.best_score)
            self.load_state_dict(early_stopping.best_checkpoint,
                                 strict=not (hasattr(self.args, 'freeze') and self.args.freeze))
        if path:
            if self.args.use_multi_gpu:
                dist.barrier()
            best_model_path = path + '_checkpoint.pth'
            print('Save checkpoint to', best_model_path)
            torch.save(self.state_dict() if self.flag_load_state_dict else self._model, best_model_path)

        return -early_stopping.best_score

    def train_step(self, loss, model_optim, iter_count):
        loss.backward()
        if iter_count % self.args.grad_accumulation == 0:
            if self.args.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
            model_optim.step()
            if model_optim is not None:
                model_optim.zero_grad()

    @torch.no_grad()
    def test(self, setting, test=0, show_progress=True, test_data=None, test_loader=None):
        if self.args.use_multi_gpu:
            self._build_ddp_model()
        self.model.eval()
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE', 'MAPE']}
        if test_data is None and self.args.gift_eval: # TODO: support gluonts wrapper
            test_data = get_test_data_from_gluonts(self.args, getattr(self, 'scaler', None))
        if test_data is None or test_loader is None:
            test_data, test_loader = self._get_data(flag='test', enlarge_bs=4)
        if show_progress:
            test_loader = tqdm(test_loader, miniters=1, mininterval=10, leave=False)
        for i, batch in enumerate(test_loader):
            batch = [d.to(self.device) for d in batch]
            batch_x, batch_y = batch[:2]
            if self.args.output_attention:
                outputs, attns = self.model(batch_x, *batch[2:])
            else:
                outputs = self.model(batch_x, *batch[2:])
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            f_dim = -1 if self.args.mode == 'MS' else 0
            pred_y = outputs[:, :self.args.pred_len, :]
            outputs = pred_y.detach().cpu()
            batch_y = batch_y.detach().cpu()

            if self.args.inverse and test_data.scale:
                shape = outputs.shape
                outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

            outputs = outputs[:, :, f_dim:]
            batch_y = batch_y[:, :, f_dim:]

            update_metrics(outputs, batch_y, statistics)

        metrics = calculate_metrics(statistics, self.device)
        mse, mae = metrics['MSE'], metrics['MAE']
        print(self.args.data_path, 'mse:{}, mae:{}'.format(mse, mae))
        return metrics


    def gift_eval(self):
        self.model.eval()
        from gift_eval.data import Dataset
        from gluonts.ev.metrics import (
            MAE,
            MAPE,
            MASE,
            MSE,
            MSIS,
            ND,
            NRMSE,
            RMSE,
            SMAPE,
            MeanWeightedSumQuantileLoss,
        )
        from gluonts.model import evaluate_model
        from gluonts.time_feature import get_seasonality

        pretty_names = {
            "saugeenday": "saugeen",
            "temperature_rain_with_missing": "temperature_rain",
            "kdd_cup_2018_with_missing": "kdd_cup_2018",
            "car_parts_with_missing": "car_parts",
        }
        ds_name = self.args.data_path
        if "/" in ds_name:
            ds_key = ds_name.split("/")[0]
            ds_freq = ds_name.split("/")[1]
            ds_key = ds_key.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
        else:
            ds_key = ds_name.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
            dataset_properties_map = json.load(open("dataset/dataset_properties.json"))
            ds_freq = dataset_properties_map[ds_key]["frequency"]
        ds_config = f"{ds_key}/{ds_freq}/{self.args.term}"
        metrics = [
            MSE(forecast_type="mean"),
            MSE(forecast_type=0.5),
            MAE(forecast_type="mean"),
            MAE(forecast_type=0.5),
            MASE(),
            MAPE(),
            SMAPE(),
            MSIS(),
            RMSE(),
            NRMSE(),
            ND(),
            MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        ]

        # Initialize the dataset
        to_univariate = (
            False
            if Dataset(name=ds_name, term=self.args.term, to_univariate=False).target_dim == 1
            else True
        )
        dataset = Dataset(name=ds_name, term=self.args.term, to_univariate=to_univariate)
        season_length = get_seasonality(dataset.freq)
        print(f"Dataset size: {len(dataset.test_data)}")

        if getattr(self, "scaler", None) is None:
            self.scaler = get_train_val_data_from_gluonts(self.args)[-1]
        self._model.scaler = self.scaler
        self._model.forecast_type = "quantile"

        # Measure the time taken for evaluation
        logging.getLogger("gluonts").setLevel(logging.INFO)
        res = evaluate_model(
            self._model,
            test_data=dataset.test_data,
            metrics=metrics,
            batch_size=1024,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length,
        )
        pd.set_option('display.max_columns', None)
        print(res[["MSE[0.5]", 'MAE[0.5]', "MAPE[0.5]", "sMAPE[0.5]", "MASE[0.5]", "mean_weighted_sum_quantile_loss",
                   "RMSE[mean]", "NRMSE[mean]"]])
        print(res.to_dict())
        return {k: v[None] for k, v in res.to_dict().items()}


    def analysis(self):
        times_infer = []
        # print('GPU Mem:', torch.cuda.max_memory_allocated())
        self.model.eval()
        self.args.batch_size = self.args.enc_in if self.args.mode == 'S' else 1
        i = 0
        test_data, test_loader = self._get_data(flag='test', enlarge_bs=1)
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                start_time = time.time()
                self.model(data[0])
                if i > 5:
                    times_infer.append(time.time() - start_time)
                if i >= 24 + 5:
                    break
        # print('Final GPU Mem:', torch.cuda.max_memory_allocated())
        times_infer = (sum(times_infer) - min(times_infer) - max(times_infer)) / (len(times_infer) - 2)
        print('Latency:', times_infer)
        # test_params_flop(self.model, (1, self.args.seq_len, self.args.enc_in))


class Exp_PEFT(Exp_Forecast):
    def _build_model(self, model=None, framework_class=None):
        if framework_class is None:
            framework_class = []
        self.args.freeze = True
        framework_class.append(importlib.import_module('layers.' + self.args.tune_method.lower()).get_peft_model)
        model = super()._build_model(framework_class=framework_class)
        return model
