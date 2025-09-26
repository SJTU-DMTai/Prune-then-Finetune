hyperparams = {
    'PatchTST': {'e_layers': 3},
    'iTransformer': {'e_layers': 3, 'd_model': 512, 'd_ff': 512, 'activation': 'gelu', 'timeenc': 1, 'patience': 3, 'train_epochs': 10, },
}

def get_hyperparams(data, model, args, reduce_bs=True):
    if model not in hyperparams:
        return None
    hyperparam: dict = hyperparams[model]
    # if model == 'iTransformer':
    #     if data == 'Traffic':
    #         hyperparam['e_layers'] = 4
    #     elif 'ETT' in data:
    #         hyperparam['e_layers'] = 2
    #         if data == 'ETTh1':
    #             hyperparam['d_model'] = 256
    #             hyperparam['d_ff'] = 256
    #         else:
    #             hyperparam['d_model'] = 128
    #             hyperparam['d_ff'] = 128

    if model == 'PatchTST':
        if args.lradj != 'type3':
            if data in ['ETTh1', 'ETTh2', 'weather', 'Weather', 'Exchange', 'wind']:
                hyperparam['lradj'] = 'type3'
            elif data in ['Illness']:
                hyperparam['lradj'] = 'constant'
            else:
                hyperparam['lradj'] = 'TST'
        if data in ['ETTh1', 'ETTh2', 'Illness']:
            hyperparam.update(**{'dropout': 0.3, 'fc_dropout': 0.3, 'n_heads': 4, 'd_model': 16, 'd_ff': 128})
        elif data in ['ETTm1', 'ETTm2', 'Weather', 'weather', 'ECL', 'Traffic']:
            hyperparam.update(**{'dropout': 0.2, 'fc_dropout': 0.2, 'n_heads': 16, 'd_model': 128, 'd_ff': 256})
        else:
            hyperparam.update(**{'dropout': 0.2, 'fc_dropout': 0.2, 'n_heads': 16, 'd_model': 64, 'd_ff': 128})
    return hyperparam