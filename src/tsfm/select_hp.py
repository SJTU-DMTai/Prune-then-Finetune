import argparse
import itertools

parser = argparse.ArgumentParser(description='Hyperparameter selection')

parser.add_argument('--lora', nargs='+', default=[],)
parser.add_argument('--prune_ratio_per_epoch', nargs='+', default=[])
parser.add_argument('--prune_ema', nargs='+', default=[],)
parser.add_argument('--head', nargs='+', default=[])
parser.add_argument('--learning_rate', nargs='+', default=[],)
parser.add_argument('--filename', type=str)

args = parser.parse_args()

results = {}
min_val, min_arg = 1e9, None

def read(hp, min_val, min_arg):
    try:
        with open(filename := args.filename.format(*hp)) as f:
            for line in list(f)[::-1]:
                if 'Best Valid MSE' in line:
                    mse = float(line.split(':')[-1].strip())
                    if mse < min_val:
                        min_val = mse
                        min_arg = hp
                    break
    except Exception as e:
        pass
    return min_val, min_arg

hps = [v for v in [args.lora, args.prune_ratio_per_epoch, args.head, args.prune_ema, args.learning_rate] if len(v)]
for hp in itertools.product(*hps):
    min_val, min_arg = read(hp, min_val, min_arg)
print(' '.join(min_arg))