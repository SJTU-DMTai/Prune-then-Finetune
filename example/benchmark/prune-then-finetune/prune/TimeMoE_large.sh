if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi

model_name=TimeMoE
model_size=large
seq_len=4096
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
  ETTh1)
    thres=0.1;;
  ETTm1|weather)
    thres=0.001;;
  *)
    thres=0.01;;
  esac
for pred_len in 96
do
  torchrun --nproc_per_node=8 src/tsfm/run.py --use_multi_gpu --prune_transformer --find_unused_parameters \
    --root_path ./datasets/$dir/ --data_path $data_path --autoregressive --valid_autoregressive \
    --model $model_name --model_size $model_size --apply_aux_loss False \
    --model_id $model_size'_prune_raw_'$thres'_pr'$prune_ratio_per_epoch'_ema'$ema'_full_shot' \
    --seq_len $seq_len --prune_expert_threshold $thres \
    --pred_len $pred_len --macro_batch_size 8192 --pruner_type taylor2 --stride $seq_len \
    --do_training 1 --wo_test --prune_ratio_per_epoch $prune_ratio_per_epoch --prune_ema $ema \
    --batch_size 16 >> ./logs/tsfm/$data'_'$model_name'_'$model_size'_L'$seq_len'_H'$pred_len'_prune_raw_'$thres'_pr'$prune_ratio_per_epoch'_ema'$ema.log 2>&1
done
done
done
done