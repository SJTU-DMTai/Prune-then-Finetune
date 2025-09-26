if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi

model_name=TimesFM
model_size=large
L=2048
for data in $1
do
for prune_ratio_per_epoch in 0.01 0.02 0.05 0.10 0.15
do
for ema in 0.1 0.2 0.4 0.5 0.8
do
for pred_len in 128
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
  python -u src/tsfm/run.py \
    --root_path ./datasets/$dir/ --data_path $data_path \
    --model $model_name --model_size $model_size --prune_transformer --finetune_epochs 1 \
    --model_id 'pr'$prune_ratio_per_epoch'_ema'$ema'_full_shot' \
    --seq_len $L \
    --pred_len $pred_len --macro_batch_size 8192 \
    --do_training 1 --wo_test --prune_ratio_per_epoch $prune_ratio_per_epoch --prune_ema $ema \
    --batch_size 64 >> ./logs/tsfm/$data'_'$model_name'_L'$L'_H'$pred_len'_pr'$prune_ratio_per_epoch'_ema'$ema.log 2>&1
done
done
done
done