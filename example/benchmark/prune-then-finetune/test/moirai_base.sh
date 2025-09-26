if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi
i=0

model_name=moirai
model_size=base
mode=S
L=3072
i=0
for data in $1
do
  case $data in
  ETTh1|ETTh2|ETTm1|ETTm2)
    dir='ETT-small'
    data_path=$data.csv;;
  PEMS03|PEMS04|PEMS07|PEMS08)
    dir='PEMS'
    data_path=$data.npz;;
  *)
    dir=$data
    data_path=$data.csv;;
  esac
  case $data in
    ETTm2)
    P=128;;
    *)
    P=64;;
  esac
  case $data in
    ETTh2|ETTm2)
    bs=64
    mode=M;;
    *)
    bs=512
    mode=S;;
  esac
  case $data in
    ETTh2|ETTm2|M-dense)
    L=4096;;
    ETTh1|ETTm1)
    L=2048;;
    *)
    L=3072;;
  esac
for pred_len in 96 192 336 720
do
#  filename=./logs/tsfm/$data'_'$model_name'_'$model_size'_'$mode'_L'$L'_H'$pred_len'_P'$P'_prune_pr{}_ema{}'.log
#  read pr ema <<< "$(python src/tsfm/select_hp.py --prune_ratio_per_epoch 0.05 0.10 0.15 0.20 --filename $filename --prune_ema 0.1 0.2 0.4 0.5 0.8)"
#  filename=./logs/tsfm/$data'_'$model_name'_'$model_size'_'$mode'_L'$L'_H'$pred_len'_P'$P'_prune_pr{}_ema{}_lr{}'.log
#  read pr ema lr <<< "$(python src/tsfm/select_hp.py --prune_ratio_per_epoch $pr --filename $filename --prune_ema $ema --learning_rate 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9)"
  filename=./logs/tsfm/$data'_'$model_name'_'$model_size'_'$mode'_L'$L'_H'$pred_len'_P'$P'_prune_pr{}_ema{}_lr{}'.log
  read pr ema lr <<< "$(python src/tsfm/select_hp.py --prune_ratio_per_epoch 0.05 0.10 0.15 0.20 --filename $filename --prune_ema 0.1 0.2 0.4 0.5 0.8 --learning_rate 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9)"
  filename=./logs/tsfm/$data'_'$model_name'_'$model_size'_'$mode'_L'$L'_H'$pred_len'_P'$P'_prune_pr'$pr'_ema'$ema'_lr'$lr.log
  if [ ! -e $filename ] || [ ! -n "$(grep 'mse:' $filename)" ]; then
  echo $filename
  CUDA_VISIBLE_DEVICES=$[i+0] python -u src/tsfm/run.py \
    --root_path ./datasets/$dir/ --data_path $data_path \
    --model $model_name --model_size $model_size --prune_transformer \
    --pruned_model_id $model_size'_'$mode'_prune_pr'$pr'_ema'$ema'_full_shot' \
    --model_id $model_size'_'$mode'_prune_pr'$pr'_ema'$ema'_full_shot_lr'$lr --learning_rate $lr \
    --seq_len $L --patch_len $P --mode $mode \
    --pred_len $pred_len \
    --do_training 1 --reload True --prune_ratio_per_epoch $pr --prune_ema $ema \
    --batch_size 2048 >> $filename 2>&1 &
  i=$[i+1]
  [ $i -eq 8 ] && wait
  i=$[i%8]
  fi
done
done