import argparse
import collections
import os
import sys

import math

from exp.exp_prune import Exp_Prune
from exp.exp_statistic import Exp_Statistics

import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from numpy.lib._iotools import str2bool

import hyparam
from exp.exp_forecast import Exp_Forecast, Exp_PEFT
from exp.exp_score import Exp_Score
from utils.tools import HiddenPrints
import wandb

if __name__ == '__main__':

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    parser = argparse.ArgumentParser(description='Post Training of Time Series Foundation Model')

    # basic config
    parser.add_argument('--task_name', type=str, default='forecast')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--pruned_model_id', type=str, default=None)
    parser.add_argument('--model', type=str, default='TimerXL')
    parser.add_argument('--model_size', type=str, default='base')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--tag', type=str, default='')

    # data loader
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--mode', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding. will be decided by ../dataset/data_info.csv')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='dirname of model checkpoints')
    parser.add_argument('--pruned_checkpoint', type=str, default=None, help='filename of pruned model checkpoint')
    parser.add_argument('--inverse', action='store_true', help='inverse output data to the raw data scale', default=False)
    parser.add_argument('--sampling_strategy', type=str, default='uniform', help='[uniform, recent] for few-shot scenarios')
    parser.add_argument('--stride', type=int, default=1, help='stride between samples in few-shot scenarios')
    parser.add_argument('--term', type=str, default="short", help='GIFT-EVAL')
    parser.add_argument('--gift_eval', action='store_true', help='indicate whether the dataset comes from GIFT-EVAL')
    parser.add_argument('--val_of_train_ratio', type=float, default=0)

    # model define
    parser.add_argument('--d_model', type=int, default=1024, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=8, help='num of encoder layers')
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--enforce_dropout', type=str2bool, default=False)
    parser.add_argument('--disable_dropout', type=str2bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--fc_dropout', type=float, default=0, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--output_activation', action='store_true', help='whether to output activation in ecoder')
    parser.add_argument('--attn_implementation', type=str, default='flash_attention_2')
    parser.add_argument('--revin', action='store_true', help='whether to use RevIN')

    # LoRA
    parser.add_argument('--tune_method', type=str, default=None)
    parser.add_argument('--lora_alpha', type=int, default=32, help='')
    parser.add_argument('--lora_rank', type=int, default=0, help='')
    parser.add_argument('--freeze', action='store_true', default=False)

    # Linear probing
    parser.add_argument('--probing', action='store_true', default=False)

    # Prune
    parser.add_argument('--prune_ratio_per_epoch', type=float, default=0, help='')
    parser.add_argument('--prune_ema', type=float, default=0,)
    parser.add_argument('--pruner_type', type=str, default='taylor2')
    parser.add_argument('--ema_abs', action='store_true')
    parser.add_argument('--prune_expert_threshold', type=float, default=0.1,)
    parser.add_argument('--prune_head', type=float, default=0,)
    parser.add_argument('--act_prob_threshold', type=float, default=0,)
    parser.add_argument('--only_prune_expert', action='store_true')
    parser.add_argument('--average_expert_score', action='store_true')
    parser.add_argument('--prune_transformer', action='store_true')
    parser.add_argument('--mask_out', action='store_true', default=True)
    parser.add_argument('--mask_in', action='store_true', default=True)
    parser.add_argument('--prune_ratio', type=float, default=0,)
    parser.add_argument('--max_prune_ratio', type=float, default=1,)
    parser.add_argument('--pruned_seq_len', type=int, default=None, help='input len when pruning')
    parser.add_argument('--pruned_pred_len', type=int, default=None, help='output len when pruning')

    # optimization
    parser.add_argument('--save_opt', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--valid_begin', type=int, default=0, help='first valid epoch')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size of train input data')
    parser.add_argument('--macro_batch_size', type=int, default=0, help='')
    parser.add_argument('--grad_accumulation', type=int, default=1)
    parser.add_argument('--patience', type=int, default=1, help='early stopping patience')
    parser.add_argument('--num_batch_per_epoch', type=float, default=0)
    parser.add_argument('--clip_grad_norm', type=float, default=0)
    parser.add_argument('--use_weight_decay', type=int, default=0, help='use weight decay')
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--val_loss', type=str, default=None)
    parser.add_argument('--apply_aux_loss', type=str2bool, default=True)
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--find_unused_parameters', action='store_true')

    # cosin decay
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing lr', default=False)
    parser.add_argument('--tmax', type=int, default=10, help='tmax in cosine anealing lr')
    parser.add_argument('--cos_warm_up_steps', type=int, default=100)
    parser.add_argument('--cos_max_decay_steps', type=int, default=60000)
    parser.add_argument('--cos_max_decay_epoch', type=int, default=10)
    parser.add_argument('--cos_max', type=float, default=1e-4)
    parser.add_argument('--cos_min', type=float, default=2e-6)
    parser.add_argument('--decay_fac', type=float, default=0.75)

    # GPU
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='use gpu')
    parser.add_argument('--pin_gpu', type=str2bool, default=True, help='move the whole dataset to gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    parser.add_argument('--subset_ratio', type=float, default=1, help='ratio of training set')
    parser.add_argument('--training_num', type=int, default=-1, help='number of training samples')

    # autoregressive configs
    parser.add_argument('--autoregressive', action='store_true', help='Autoregressive', default=False)
    parser.add_argument('--valid_autoregressive', action='store_true', help='Autoregressive', default=False)

    # train_test
    parser.add_argument('--train_test', type=int, default=0, help='train_test')
    parser.add_argument('--do_training', type=int, default=1, help='status')
    parser.add_argument('--finetune_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--wo_test', action='store_true', default=False)
    parser.add_argument('--only_valid', action='store_true', default=False)
    parser.add_argument('--save_per_epoch', action='store_true', default=False)
    parser.add_argument('--load_epoch_id', type=str, default='')
    parser.add_argument('--reload', type=str2bool, default=False)
    parser.add_argument('--test_speed', action='store_true', default=False)

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=672, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=576, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='output len')
    parser.add_argument('--patch_len', type=int, default=64, help='input sequence length')
    parser.add_argument('--input_ensemble', action='store_true', default=False)

    args = parser.parse_args()
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.test_speed:
        args.use_gpu = False
        args.attn_implementation = "eager"

    if args.grad_accumulation > 1:
        args.batch_size //= args.grad_accumulation

    if args.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "1"))  # number of nodes
        rank = int(os.environ.get("RANK", "0"))  # node id
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()  # gpus per node
        args.batch_size = args.batch_size // hosts
        args.local_rank = local_rank
        print(
            'ip: {}, port: {}, hosts: {}, rank: {}, local_rank: {}, gpus: {}'.format(ip, port, hosts, rank, local_rank,
                                                                                     gpus))
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts, rank=rank)
        print('init_process_group finished')
        torch.cuda.set_device(local_rank)


    hyperparams = hyparam.get_hyperparams(args.data_path.split('.')[0], args.model, args)
    if hyperparams:
        for k, v in hyperparams.items():
            args.__setattr__(k, v)

    if args.gift_eval:
        from gift_eval.data import Dataset
        args.val_loss = 'mae'
        args.to_univariate = (
            False
            if Dataset(name=args.data_path, term=args.term, to_univariate=False).target_dim == 1
            else True
        )
        dataset = Dataset(name=args.data_path, term=args.term, to_univariate=args.to_univariate)
        all_lengths = []
        for x in dataset.training_dataset:
            if len(x["target"].shape) == 1:
                all_lengths.append(len(x["target"]))
                num_channels = 1
            else:
                all_lengths.append(x["target"].shape[1])
                num_channels = x["target"].shape[0]
        args.target_dim = num_channels
        args.freq = dataset.freq
        args.scale = True
        if args.model == 'TTM':
            min_context_length = min(all_lengths)
            print(
                "Minimum context length among all time series in this dataset =",
                min_context_length,
            )
            args.seq_len = min(args.seq_len, min_context_length)
        elif max(all_lengths) - dataset.prediction_length < args.seq_len:
            args.seq_len = math.ceil((max(all_lengths) - dataset.prediction_length) / 16) * 16 # Assume the patch size is 16
        print(args.seq_len)
        args.pred_len = dataset.prediction_length
        args.pin_gpu = False
        args.num_workers = max(args.num_workers, 4)
    else:
        info_table = pd.read_csv("dataset/data_info.csv", index_col=0).to_dict()
        args.target_dim = args.enc_in = info_table["dim"][args.data_path.split('.')[0]]
        args.freq = info_table["freq"][args.data_path.split('.')[0]]
    if 'timer' in args.model.lower() or 'chronos' in args.model.lower():
        args.mode = 'S'
        args.patch_len = 96

    if args.model == 'TimeMoE':
        args.mode = 'S'
        args.dtype = torch.bfloat16
        args.patch_len = 1

    if args.autoregressive:
        args.label_len = args.seq_len - args.patch_len
    else:
        args.label_len = 0

    if args.num_workers > 0:
        args.pin_gpu = False

    if args.task_name == 'score':
        Exp = Exp_Score
        args.prune_transformer = True
        args.reload = True
    elif args.task_name == 'prune' or args.prune_ratio_per_epoch or args.only_prune_expert:
        Exp = Exp_Prune
        args.task_name = 'forecast'
    elif args.pruner_type == 'norm' or args.act_prob_threshold or args.prune_head:
        Exp = Exp_Statistics
    elif args.tune_method or args.lora_rank:
        Exp = Exp_PEFT
    else:
        Exp = Exp_Forecast

    if 'Chronos' in args.model:
        trained_pred_len = 64
    elif 'TimesFM' in args.model:
        trained_pred_len = 128
    elif 'TimerXL' in args.model:
        trained_pred_len = 96
    elif 'TimeMoE' in args.model:
        trained_pred_len = 96
    else:
        trained_pred_len = args.pred_len

    wandb.init(project='TSFM',
               name=f"{args.tag}{args.data_path.replace('/', '_').split('.')[0]}_{args.model}_{args.model_id}_H{args.pred_len}")
    wandb.config.update(args)
    print('Args in experiment:')
    print(args)
    all_metrics = collections.defaultdict(list)
    with HiddenPrints(int(os.environ.get("LOCAL_RANK", "0"))):
        for ii in range(args.itr):
            exp = Exp(args)
            # setting record of experiments
            setting = '{}_{}_{}_{}_sl{}_pl{}_ps{}_{}'.format(
                args.task_name if args.task_name != 'score' else 'forecast',
                args.model,
                args.model_id,
                args.data_path.replace('/', '_').split('.')[0],
                args.seq_len if args.model != 'TimeMoE' else 4096,
                trained_pred_len,
                args.patch_len,
                ii)

            if isinstance(exp, Exp_Statistics):
                exp.merge_weights(setting)

            if args.pruned_checkpoint:
                print('Loading', os.path.join(args.checkpoints, args.pruned_checkpoint))
                exp.load_checkpoint(os.path.join(args.checkpoints, args.pruned_checkpoint), strict=False)
                exp.merge_weights()
            elif args.pruned_model_id:
                assert 'lr' in args.model_id
                setting_pruned = '{}_{}_{}_{}_sl{}_pl{}_ps{}_{}'.format(
                    args.task_name if args.task_name != 'score' else 'forecast',
                    args.model,
                    args.pruned_model_id,
                    args.data_path.replace('/', '_').split('.')[0],
                    (args.pruned_seq_len or args.seq_len) if args.model != 'TimeMoE' else 4096,
                    trained_pred_len,
                    args.patch_len,
                    ii)
                path = os.path.join(args.checkpoints, setting_pruned + '_checkpoint.pth' + args.load_epoch_id)
                print('Loading', path)
                exp.load_checkpoint(path, strict=False)
                exp.merge_weights(setting_pruned)

            path = os.path.join(args.checkpoints, setting + '_checkpoint.pth')
            path.replace('_bestpr', '')
            path.replace('_best', '')
            if args.reload and os.path.exists(path):
                print('Loading', path)
                exp.load_checkpoint(path, strict=False)
                if args.test_speed:
                    exp.analysis()
                    exit(0)
                if isinstance(exp, Exp_Prune) and args.pruned_model_id is None and args.pruned_checkpoint is None:
                    exp.merge_weights(setting)
                if args.task_name == 'score':
                    exp.train(setting)
            else:
                if args.do_training and args.learning_rate >= 0:
                    # if args.reload and not os.path.exists(path):
                    #     print('Not exist ' + path)
                    #     assert not isinstance(exp, Exp_Prune)
                    print('Checkpoint will be saved to', path)
                    if isinstance(exp, Exp_Prune) and args.pruned_model_id is None:
                        print('>>>>>>>start pruning : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                        val_metric = exp.prune(setting)
                        if args.only_prune_expert:
                            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                            val_metric = exp.train(setting)
                        wandb.summary.update({'val_metric': val_metric})
                    else:
                        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                        val_metric = exp.train(setting)
                        wandb.summary.update({'val_metric': val_metric})

            if args.test_speed:
                exp.analysis()
            elif args.only_valid:
                while True:
                    try:
                        print('Best Valid MSE:', exp.vali())
                        break
                    except torch.OutOfMemoryError:
                        print(
                            f"OutOfMemoryError at batch_size {args.batch_size}, reducing to {args.batch_size//2}"
                        )
                        args.batch_size //= 2
            elif not args.wo_test:
                print('Model parameters: ', sum(param.numel() for param in exp.model.parameters()))
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                while True:
                    try:
                        if 'pretrain' in args.task_name:
                            exp.ood_test(setting)
                        else:
                            metrics = {}
                            if int(os.environ.get("LOCAL_RANK", "0")) <= 0:
                                if not args.gift_eval:
                                    metrics = exp.test(setting)
                                else:
                                    metrics = exp.gift_eval()
                            for k, v in metrics.items():
                                all_metrics[k].append(v)
                            torch.cuda.empty_cache()
                        break
                    except torch.OutOfMemoryError:
                        print(
                            f"OutOfMemoryError at batch_size {args.batch_size}, reducing to {args.batch_size//2}"
                        )
                        args.batch_size //= 2
    for k, v in all_metrics.items():
        for i, vv in enumerate(v):
            wandb.summary.update({f"{k}_{i}": vv})
    wandb.summary.update(all_metrics)
    all_metrics = {k: np.array(v).mean() for k, v in all_metrics.items()}
    wandb.summary.update(all_metrics)
