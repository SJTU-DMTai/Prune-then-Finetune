import copy
import math
import os
import numpy as np
import pandas as pd
from typing import List
from gluonts.dataset.split import InputDataset, LabelDataset, TrainingDataset
from torch.utils.data import ConcatDataset, Subset

from data_provider.gluonts_data_wrapper import StandardScalingGluonTSDataset, TorchDatasetFromGluonTSTrainingDataset, \
    TorchDatasetFromGluonTSTestDataset


def process_time_series(dataset: TrainingDataset, truncate: bool = True, past_feat_dynamic_real_exist: bool = False) -> List:
    """
    Processes a time series by truncating initial NaNs and forward filling intermittent NaNs.
    Returns a new truncated dataset, and does not modify the original one.

    Args:
        dataset (TrainingDataset): Every series of of shape [channels, length].
        truncate (bool, optional): Truncate the dataset if True. Defaults to True.

    Returns:
        List: Processed time series, each of shape [channels, truncated_length].
    """
    truncated_dataset = list(copy.deepcopy(dataset))
    for i, item in enumerate(truncated_dataset):
        data = item["target"]

        if data.ndim == 1:
            data = data.reshape(1, -1)  # [channels, length]

        if past_feat_dynamic_real_exist:
            if item["past_feat_dynamic_real"].ndim == 1:
                item["past_feat_dynamic_real"] = item["past_feat_dynamic_real"].reshape(1, -1)
            data = np.vstack((data, item["past_feat_dynamic_real"]))

        truncated_dataset[i]["target"] = data

        if not truncate:
            continue

        # Step 1: Determine the longest stretch of initial NaNs across all channels
        valid_mask = ~np.isnan(data)  # Mask of valid (non-NaN) values

        if valid_mask.all():
            continue  # Continue if no NaN

        first_valid = np.argmax(valid_mask.any(axis=0))  # First col with any valid value across channels
        data = data[:, first_valid:]  # Truncate cols before the first valid col

        # Step 2: Perform forward fill for NaNs
        df = pd.DataFrame(data.T, columns=range(data.shape[0]))
        df = df.ffill(axis=0)

        data = df.values.T
        if data.shape[0] == 1:  # [1, truncated_length]
            data = data.reshape(-1)  # [truncated_length]

        truncated_dataset[i]["target"] = data

    return truncated_dataset


def get_train_val_data_from_gluonts(args):
    from gift_eval.data import Dataset
    dataset = Dataset(name=args.data_path, term=args.term, to_univariate=args.to_univariate)
    train_dataset = dataset.training_dataset
    valid_dataset = dataset.validation_dataset
    train_dataset_scaled = process_time_series(train_dataset)
    valid_dataset_scaled = process_time_series(valid_dataset)
    print(f"Number of series: Train = {len(train_dataset_scaled)}, Valid = {len(valid_dataset_scaled)}")
    if getattr(args, "scale", False):
        print("Scaling training data")
        scaler = StandardScalingGluonTSDataset()
        scaler.fit(train_dataset_scaled)
        train_dataset_scaled = scaler.transform(train_dataset_scaled)
        valid_dataset_scaled = scaler.transform(valid_dataset_scaled)
    else:
        scaler = None

    # create train dataset
    dset_train = TorchDatasetFromGluonTSTrainingDataset(
        train_dataset_scaled,
        args.seq_len,
        args.pred_len,
        force_short_context=getattr(args, "force_short_context", False),
        min_context_mult=getattr(args, "min_context_mult", 4),
        send_freq=getattr(args, "enable_prefix_tuning", False),
        freq=args.freq,
        use_mask=getattr(args, "use_mask", True),
    )
    dset_valid = TorchDatasetFromGluonTSTrainingDataset(
        valid_dataset_scaled,
        args.seq_len,
        args.pred_len,
        gen_more_samples_for_short_series=False,
        force_short_context=getattr(args, "force_short_context", False),
        min_context_mult=getattr(args, "min_context_mult", 4),
        send_freq=getattr(args, "enable_prefix_tuning", False),
        freq=args.freq,
        use_mask=getattr(args, "use_mask", True),
    )
    print(f"#train: {len(dset_train)}, #valid: {len(dset_valid)}")
    if len(dset_valid) / (len(dset_valid) + len(dset_train)) < getattr(args, "val_of_train_ratio", 0):
        all_indx = list(range(0, len(dset_train)))
        val_num = int((len(all_indx) + len(dset_valid)) * args.val_of_train_ratio) - len(dset_valid)
        train_num = len(all_indx) + len(dset_valid) - val_num
        dset_valid = ConcatDataset([Subset(dset_train, all_indx[train_num:]), dset_valid])
        dset_train = Subset(dset_train, all_indx[:train_num])
        print(f"#train: {len(dset_train)}, #valid: {len(dset_valid)}")
    return dset_train, dset_valid, scaler

# TODO: Not tested yet
def get_test_data_from_gluonts(args, scaler=None):
    from gift_eval.data import Dataset
    dataset = Dataset(name=args.data_path, term=args.term, to_univariate=False)
    test_dataset = dataset.test_dataset
    test_dataset_scaled = process_time_series(test_dataset)
    dset_test = TorchDatasetFromGluonTSTestDataset(test_dataset_scaled.input, test_dataset_scaled.label,
                                                   context_length=args.seq_len, prediction_length=args.pred_len,
                                                   force_short_context=getattr(args, "force_short_context", False),
                                                   min_context_mult=getattr(args, "min_context_mult", 4),
                                                   use_mask=getattr(args, "use_mask", False),)
    print(f"Number of series: Test = {len(test_dataset_scaled)}")
    if scaler is None:
        train_dataset = dataset.train_dataset
        train_dataset_scaled = process_time_series(train_dataset)
        scaler = StandardScalingGluonTSDataset()
        scaler.fit(train_dataset_scaled)
    dset_test = scaler.transform(dset_test)
    return dset_test