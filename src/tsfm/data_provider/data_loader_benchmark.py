import os

import math
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


class CIDatasetBenchmark(Dataset):
    def __init__(self, root_path='dataset', flag='train', input_len=None, pred_len=None, label_len=None,
                 scale=True, timeenc=1, freq='h', stride=1,
                 subset_ratio=1.0, sampling_strategy='uniform', training_num=-1, **kwargs):
        self.subset_ratio = subset_ratio
        # size [seq_len, label_len, pred_len]
        # info
        self.input_len = input_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.seq_len = input_len + pred_len
        self.timeenc = timeenc
        self.scale = scale
        self.stride = stride
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.sampling_strategy = sampling_strategy
        if self.sampling_strategy == 'uniform':
            if self.set_type == 0:
                self.internal = int(1 // self.subset_ratio)
            else:
                self.internal = 1

        self.root_path = root_path
        self.dataset_name = self.root_path.split('/')[-1].split('.')[0]

        self.__confirm_data__()

        if self.sampling_strategy == 'recent' and flag == 'train':
            if training_num > 0:
                recent_timestamp_num = training_num + self.input_len + self.pred_len - 1
            else:
                recent_timestamp_num = math.ceil(len(self.data_x) * self.subset_ratio)
            self.data_x = self.data_x[-recent_timestamp_num:]
            self.data_y = self.data_y[-recent_timestamp_num:]
            self.data_stamp = self.data_stamp[-recent_timestamp_num:]
            self.n_timepoint = (len(self.data_x) - self.input_len - self.pred_len) // self.stride + 1

    def __read_data__(self):
        dataset_file_path = self.root_path
        if dataset_file_path.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file_path)
        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)

        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))

        if 'etth' in self.root_path.lower():
            border1s = [0, 12 * 30 * 24 - self.input_len, 12 * 30 * 24 + 4 * 30 * 24 - self.input_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif 'ettm' in self.root_path.lower():
            border1s = [0, 12 * 30 * 24 * 4 - self.input_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.input_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        elif 'pems' in self.root_path.lower():
            data_len = len(df_raw)
            num_train = int(data_len * 0.6)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.input_len, data_len - num_test - self.input_len]
            border2s = [num_train, num_train + num_vali, data_len]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.input_len, data_len - num_test - self.input_len]
            border2s = [num_train, num_train + num_vali, data_len]

        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values

        if self.scale:
            self.scaler = StandardScaler()
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        if self.timeenc == 0:
            df_stamp = df_raw[['date']]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            if isinstance(df_raw[df_raw.columns[0]][2], str):
                data_stamp = time_features(pd.to_datetime(pd.to_datetime(df_raw.date).values), freq='h')
                data_stamp = data_stamp.transpose(1, 0)
            else:
                data_stamp = np.zeros((len(df_raw), 4))
        else:
            raise ValueError('Unknown timeenc: {}'.format(self.timeenc))

        return data, data_stamp, border1s, border2s

    def __confirm_data__(self):
        data, data_stamp, border1s, border2s = self.__read_data__()
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.data_x = torch.tensor(data[border1:border2])
        self.data_y = torch.tensor(data[border1:border2])
        self.data_stamp = torch.tensor(data_stamp[border1:border2])

        self.n_var = self.data_x.shape[-1]

        self.n_timepoint = (len(self.data_x) - self.input_len - self.pred_len) // self.stride + 1

    def _getid(self, index):
        if self.set_type == 0 and self.sampling_strategy == 'uniform':
            index = index * self.internal
        c_begin = index // self.n_timepoint                  # select variable
        s_begin = (index % self.n_timepoint) * self.stride   # select start time
        return s_begin, c_begin

    def _getitem(self, s_begin, c_begin ):
        s_end = s_begin + self.input_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, c_begin:c_begin + 1]
        seq_y = self.data_y[r_begin:r_end, c_begin:c_begin + 1]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y
        # return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __getitem__(self, index):
        return self._getitem(*self._getid(index))

    def __len__(self):
        if self.set_type == 0 and self.sampling_strategy == 'uniform':
            return max(int(self.n_var * self.n_timepoint * self.subset_ratio), 1)
        else:
            return int(self.n_var * self.n_timepoint)

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            if not hasattr(self, 'mean'):
                self.mean = torch.tensor(self.scaler.mean_, device=data.device)
                self.std = torch.tensor(self.scaler.scale_, device=data.device)
            return self.mean + data * self.std
        else:
            return self.scaler.inverse_transform(data)

class CDDatasetBenchmark(CIDatasetBenchmark):
    def __len__(self):
        if self.set_type == 0 and self.sampling_strategy == 'uniform':
            return max(int(self.n_timepoint * self.subset_ratio), 1)
        else:
            return int(self.n_timepoint)

    def __getitem__(self, index):
        index *= self.stride
        if self.set_type == 0 and self.sampling_strategy == 'uniform':
            index = index * self.internal
        s_end = index + self.input_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[index:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y

class CIDatasetBenchmarkCtx(CIDatasetBenchmark):
    def __init__(self, freq_id, domain_id, period,
                 root_path='dataset', flag='train', input_len=None, pred_len=None, label_len=None, scale=True,
                 timeenc=1, freq='h', stride=1, subset_ratio=1.0, sampling_strategy='uniform', training_num=-1,
                 ctx_num=8, temperature=1., **kwargs):
        self.freq_id = freq_id
        self.domain_id = domain_id
        self.ctx_num = ctx_num
        self.period = period
        self.temperature = math.sqrt(period)
        self.min_pred_len = 96
        self.margin = pred_len + self.ctx_num - 1
        self.context_len = input_len + pred_len + self.margin
        self.same_prob = torch.ones(self.context_len * 2)
        self.sampling_distribution = None
        super().__init__(root_path, flag, input_len, pred_len, label_len, scale, timeenc, freq, stride, subset_ratio,
                         sampling_strategy, training_num)
        self.n_timepoint -= self.margin

    def __read_data__(self):
        data, data_stamp, border1s, border2s = super().__read_data__()
        border1s[self.set_type] = max(0, border1s[self.set_type] - self.margin)

        if self.period > 1:
            suffix_len = len(self.root_path.split('.')[-1]) + 1
            prob_path = self.root_path[:-suffix_len] + f"_sampling_distribution_{self.min_pred_len}.npy"
            if os.path.exists(prob_path):
                self.sampling_distribution = np.load(prob_path)

            if (self.sampling_distribution is None or self.sampling_distribution.shape[0] != self.period or
                    self.sampling_distribution.shape[1] != data.shape[-1]):
                val_data = data[:border2s[1]][-1000 - self.period * 2 - self.input_len - self.pred_len + 1:]
                try:
                    dist = self.estimate_sim(val_data, self.period).cpu().numpy()
                except Exception as e:
                    dist = self.estimate_sim(val_data, self.period, 'cpu').numpy()
                # if (dist[0] > 0.1).any():
                #     print(self.root_path, 'some non-periodic!')
                self.sampling_distribution = dist
                # self.sampling_distribution = self.sampling_distribution / self.sampling_distribution.sum(-1, keepdims=True)
                np.save(prob_path, self.sampling_distribution)

        # print(self.sampling_distribution.mean(-1))
        self.sampling_distribution = torch.tensor(self.sampling_distribution)
        return data, data_stamp, border1s, border2s

    def estimate_sim(self, data, period, device='cuda'):
        _idx = (torch.arange(len(data) - self.input_len - self.pred_len + 1, device=device).unsqueeze(-1) +
                torch.arange(self.input_len + self.pred_len, device=device))
        samples = torch.tensor(data, device=device)[_idx]
        samples -= samples.mean(1, keepdim=True)
        samples /= torch.sqrt(torch.var(samples, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # dist = [torch.ones(data.shape[-1], device=device)]
        dist = []
        for j in tqdm(range(self.min_pred_len, self.min_pred_len + period), disable=period <= 300):
            dist.append((samples[j:] * samples[:-j]).mean(dim=1).mean(0))
        dist = torch.softmax(torch.vstack(dist), 0)
        assert not torch.isnan(dist).any()
        return dist

    def __getitem__(self, index):
        s_begin, c_begin = self._getid(index)
        s_begin += self.margin
        ret = self._getitem(s_begin, c_begin)
        total_condidate = s_begin - self.pred_len + 1
        sample_prob = self.sampling_distribution
        # replace = total_condidate < self.ctx_num # Always False
        if sample_prob is None or (total_condidate < self.period and sample_prob[:total_condidate].sum() <= 0):
            ctx_idx = torch.multinomial(self.same_prob[:total_condidate], self.ctx_num, replacement=False)
        else:
            sample_prob = sample_prob[:, c_begin]
            period = len(sample_prob)
            if total_condidate < period:
                ctx_idx = torch.multinomial(sample_prob[:total_condidate], self.ctx_num, replacement=False)
            else:
                ctx_idx = torch.multinomial(sample_prob[:period], self.ctx_num, replacement=False) + \
                          period * torch.randint(total_condidate//period, (self.ctx_num, ), device=self.data_x.device)

        ctx_idx = s_begin - self.pred_len - ctx_idx
        # ctx_idx = torch.arange(self.ctx_num, device=self.data_x.device)
        # ctx_idx = torch.zeros(self.ctx_num, device=self.data_x.device, dtype=torch.long)
        ctx_x, ctx_y = zip(*tuple(self._getitem(i, c_begin)[:2] for i in ctx_idx))
        return *ret, torch.stack(ctx_x), torch.stack(ctx_y), self.domain_id, self.freq_id
