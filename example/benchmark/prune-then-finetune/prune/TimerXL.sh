if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi

model_name=TimerXL
seq_len=2880
for data in $1
do
for prune_ratio_per_epoch in 0.05 0.10 0.15 0.20
do
for ema in 0.1 0.2 0.4 0.5 0.8
do
for pred_len in 96
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
    --model $model_name \
    --model_id 'prune_pr'$prune_ratio_per_epoch'_ema'$ema'_full_shot' \
    --seq_len $seq_len --macro_batch_size 8192 \
    --pred_len $pred_len --stride $seq_len \
    --do_training 1 --wo_test --prune_ratio_per_epoch $prune_ratio_per_epoch --prune_ema $ema --autoregressive \
    --batch_size 512 >> ./logs/tsfm/$data'_'$model_name'_L'$seq_len'_H'$pred_len'_prune_pr'$prune_ratio_per_epoch'_ema'$ema.log 2>&1
done
done
done
done