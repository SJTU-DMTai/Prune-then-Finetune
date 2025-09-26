import bisect
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from data_provider.data_loader import UTSD_Npy, UTSD_Npy_Ctx
from data_provider.data_loader_benchmark import CIDatasetBenchmark, CIDatasetBenchmarkCtx, CDDatasetBenchmark


data_paths = [
    'ETT-small/ETTh1.csv',
    'ETT-small/ETTh2.csv',
    'ETT-small/ETTm1.csv',
    'ETT-small/ETTm2.csv',
    'electricity/electricity.csv',
    'weather/weather.csv',
    'traffic/traffic.csv',
    'PEMS/PEMS03.npz',
    'PEMS/PEMS04.npz',
    'Solar/solar_AL.txt',
]

def get_domain_freq_period(path):
    if 'ett' in path or 'electricity' in path or 'ecl' in path:
        domain = 'Energy'
        if 'ettm' in path:
            freq = 15 * 60; period = 24 * 4
        else:
            freq = 60 * 60; period = 24
    elif 'traffic' in path:
        domain = 'Transport'; freq = 60 * 60; period = 24
    elif 'pems' in path:
        domain = 'Transport'; freq = 5 * 60; period = 12 * 24
    elif 'weather' in path:
        domain = 'Nature'
        freq = 60 * 10; period = 24 * 6
    elif 'solar' in path:
        domain = 'Nature'; freq = 60 * 10; period = 24 * 6
    else:
        domain = 'Unknown'; freq = 0; period = 1
    return domain, freq, period


def data_provider(args, flag, **kwargs):
    timeenc = 0 if args.embed != 'timeF' else 1
    pred_len = kwargs.get('pred_len', None) or (args.patch_len
                                                if args.autoregressive and flag == 'train'
                                                   or args.valid_autoregressive and flag == 'val'
                                                else args.pred_len)

    if 'pretrain' in args.task_name:
        root_path = args.root_path
        if getattr(args, 'use_metadata', False):
            data_cls = UTSD_Npy_Ctx
            size = (args.seq_len, args.input_token_len, args.patch_len)
            if flag != 'train':
                kwargs.update(domain2id=args.DOMAIN2ID)
                kwargs.update(freq2id=args.FREQ2ID)
        else:
            data_cls = UTSD_Npy
            size = (args.seq_len, args.patch_len, args.patch_len)
        kwargs.update(size=size)
    else:
        root_path = os.path.join(args.root_path, args.data_path)
        if getattr(args, 'use_metadata', False):
            data_cls = CIDatasetBenchmarkCtx
            domain, freq, period = get_domain_freq_period(root_path.lower())
            kwargs.update(period=period)
            kwargs.update(domain_id=args.DOMAIN2ID[domain] if domain in args.DOMAIN2ID else 0)
            if freq in args.FREQ2ID:
                kwargs.update(freq_id=args.FREQ2ID[freq])
            else:
                # FREQs = list(sorted(args.FREQ2ID.keys()))
                # for i, _freq in enumerate(FREQs):
                #     if _freq > freq:
                #         break
                # freq_id = max(0, i - 1)
                # if freq_id < len(FREQ2ID) - 1:
                #     freq_id = (freq - FREQs[i - 1]) / (FREQs[i] - FREQs[i - 1]) + i - 1
                kwargs.update(freq_id=0)
        else:
            data_cls = CIDatasetBenchmark if args.mode == 'S' and args.model not in ['DUET', 'PatchTST'] else CDDatasetBenchmark
    # else:
    #     raise NotImplementedError

    label_len = kwargs.get('label_len', args.label_len if flag == 'train' else 0)

    data_set = data_cls(
        root_path=root_path,
        flag=flag,
        input_len=args.seq_len,
        label_len=label_len,
        pred_len=pred_len,
        scale=True,
        timeenc=timeenc,
        freq=args.freq,
        stride=args.stride if flag == 'train' or flag == 'val' and args.valid_autoregressive else 1,
        subset_ratio=args.subset_ratio,
        sampling_strategy=args.sampling_strategy,
        training_num=args.training_num,
    )
    if hasattr(data_set, 'data_x'):
        if args.pin_gpu:
            data_set.data_x = torch.tensor(data_set.data_x, dtype=getattr(args, 'dtype', torch.float32), device=args.device)
            data_set.data_y = torch.tensor(data_set.data_y, dtype=getattr(args, 'dtype', torch.float32), device=args.device)
            if hasattr(data_set, 'sampling_distribution') and data_set.sampling_distribution is not None:
                data_set.sampling_distribution = torch.tensor(data_set.sampling_distribution, dtype=getattr(args, 'dtype', torch.float32), device=args.device)
                data_set.same_prob = torch.tensor(data_set.same_prob, dtype=getattr(args, 'dtype', torch.float32), device=args.device)
        else:
            data_set.data_x = torch.tensor(data_set.data_x, dtype=getattr(args, 'dtype', torch.float32))
            data_set.data_y = torch.tensor(data_set.data_y, dtype=getattr(args, 'dtype', torch.float32))

    print(flag, len(data_set))
    return data_set, get_dataloader(data_set, args, flag, **kwargs)


def get_dataloader(data_set, args, flag, **kwargs):
    shuffle_flag = flag == 'train'
    drop_last = 'pretrain' in args.task_name
    if args.use_multi_gpu and (flag != 'test' or args.reload):
        train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(data_set,
                                 batch_size=args.batch_size,
                                 sampler=train_datasampler,
                                 num_workers=args.num_workers,
                                 persistent_workers=args.num_workers > 0,
                                 drop_last=drop_last,
                                 prefetch_factor=4 if args.task_name == 'pretrain' else None,
                                 pin_memory=args.num_workers > 0,
                                 )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            persistent_workers='pretrain' in args.task_name and args.num_workers > 0,
            prefetch_factor=2 if not args.pin_gpu else None,
            pin_memory=args.num_workers > 0,
            drop_last=drop_last,)
    return data_loader