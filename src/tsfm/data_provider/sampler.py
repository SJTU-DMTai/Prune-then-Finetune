from operator import itemgetter

import math
import numpy as np
from torch.utils.data import DistributedSampler, ConcatDataset
from torch.utils.data.distributed import Iterator

class DistributedBalancedSampler(DistributedSampler):
    def __init__(self, dataset: ConcatDataset, sizes: np.ndarray[int], flag_val: bool = False, max_len: int = 2**25,
                 *args, **kwargs) -> None:
        if flag_val:
            kwargs.update(drop_last=False)
        super().__init__(dataset, *args, **kwargs)
        self.len_dataset = min(len(dataset), max_len)
        if self.drop_last and self.len_dataset % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self.len_dataset - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self.len_dataset / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.sizes = sizes
        self.dataset_sizes = [len(d) for d in dataset.datasets]
        self.cumulative_sizes = [0] + dataset.cumulative_sizes
        self.probability = np.array(sizes / sizes.sum(), dtype=np.float32)
        self.flag_val = flag_val
        if flag_val:
            # min_size = min(self.sizes) * factor
            g = np.random.default_rng(self.seed)
            indices = []
            for i, size in enumerate(self.sizes):
                assert size <= self.dataset_sizes[i]
                if size == self.dataset_sizes[i]:
                    indices.append(np.arange(self.cumulative_sizes[i], self.cumulative_sizes[i] + size))
                else:
                    indices.append(self.cumulative_sizes[i] +
                                   g.choice(self.dataset_sizes[i], size, replace=False))
            self.indices = np.concatenate(indices).tolist()
            self.drop_last = False
            self.num_samples = math.ceil(len(self.indices) / self.num_replicas)  # type: ignore[arg-type]
            self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator:
        if self.flag_val:
            return iter(self.indices[self.rank:self.total_size:self.num_replicas])

        # deterministically shuffle based on epoch and seed
        g = np.random.default_rng(self.seed + self.epoch)
        indices = g.choice(len(self.dataset_sizes), self.len_dataset, replace=True, p=self.probability)

        for i, size in enumerate(self.dataset_sizes):
            mask = indices == i
            num = mask.sum().item()
            fill = [self.cumulative_sizes[i] + g.permutation(size) for _ in range(num // size)]
            pad = num % size
            if pad:
                fill.append(self.cumulative_sizes[i] + g.choice(size, pad, replace=False))
            indices[mask] = np.concatenate(fill)
        indices = indices.tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
