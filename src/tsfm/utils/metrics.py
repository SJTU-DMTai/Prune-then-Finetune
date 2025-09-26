import os

import numpy as np
import torch
from torch import distributed as dist


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return ((pred - true) / (true + 1e-9)).abs().mean()


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def NRMSE(pred, true):
    return np.sqrt(MSE(pred, true)) / np.mean(np.abs(true))


def WAPE(pred, true):
    return np.mean(np.abs(pred - true)) / np.mean(np.abs(true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def update_metrics(pred, label, statistics, target_variate=None):
    if isinstance(pred, tuple):
        pred = pred[0]
    if target_variate is not None:
        pred = pred[:, :, target_variate]
        if label.dim() == 3:
            label = label[:, :, target_variate]

    balance = pred - label
    # statistics['all_preds'].append(pred)
    statistics['y_sum'] += label.abs().sum().item()
    statistics['total'] += len(label.flatten())
    statistics['MAE'] += balance.abs().sum().item()
    statistics['MSE'] += (balance ** 2).sum().item()
    statistics['MAPE'] += (balance / label).abs().sum().item()
    # RRSE += (balance ** 2).sum()
    # x2_sum += (target_batch ** 2).sum()
    # x_sum += target_batch.sum()


def calculate_metrics(statistics, device=None):
    MSE, MAE, total, y_sum = statistics['MSE'], statistics['MAE'], statistics['total'], statistics['y_sum']
    MAPE = statistics['MAPE']
    if int(os.getenv("LOCAL_RANK", "-1")) > 0:
        MSE, MAE, total = torch.tensor(MSE, device=device), torch.tensor(MAE, device=device), torch.tensor(total, device=device)
        MAPE = torch.tensor(MAPE, device=device)
        dist.all_reduce(MSE, op=dist.ReduceOp.SUM)
        dist.all_reduce(MAE, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(MAPE, op=dist.ReduceOp.SUM)
        MSE, MAE, total = MSE.item(), MAE.item(), total.item()
        MAPE = MAPE.item()
    metrics = {'MSE': MSE / total, 'MAE': MAE / total, 'MAPE': MAPE / total}
    # metrics['NMAE'] = MAE / y_sum
    # metrics['NRMSE'] = math.sqrt((MSE / total)) / (y_sum / total)
    # var = x2_sum / total - (x_sum / total) ** 2
    # RRSE = math.sqrt(RRSE.item() / total) / var.item()
    return metrics
