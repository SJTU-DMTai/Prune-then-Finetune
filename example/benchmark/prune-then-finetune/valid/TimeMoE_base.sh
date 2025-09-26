if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi

model_name=TimeMoE
model_size=base
seq_len=3072
for data in $1
do
for prune_ratio_per_epoch in 0.01 0.02 0.05 0.10
do
for ema in 0.1 0.2 0.4 0.5 0.8
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
  ETTh2)
    lr=1e-5;;
  *)
    lr=1e-4;;
  esac
  case $data in
  ETTh1)
    thres=0.1;;
  ETTm1)
    thres=0.001;;
  *)
    thres=0.01;;
  esac
for pred_len in 96
do
  python -u src/tsfm/run.py \
    --root_path ./datasets/$dir/ --data_path $data_path --autoregressive \
    --model $model_name --model_size $model_size --apply_aux_loss False --prune_transformer \
    --model_id $model_size'_prune_raw_'$thres'_pr'$prune_ratio_per_epoch'_ema'$ema'_full_shot' \
    --seq_len $seq_len --prune_expert_threshold $thres --stride $seq_len \
    --pred_len $pred_len --pruner_type taylor2 --prune_transformer \
    --do_training 0 --only_valid --reload True --prune_ratio_per_epoch $prune_ratio_per_epoch --prune_ema $ema \
    --batch_size 32 >> ./logs/tsfm/$data'_'$model_name'_'$model_size'_L'$seq_len'_H'$pred_len'_prune_raw_'$thres'_pr'$prune_ratio_per_epoch'_ema'$ema.log 2>&1
done
done
done
done