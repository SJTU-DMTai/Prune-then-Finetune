if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi

model_name=TimeMoE
model_size=base
seq_len=4096
for data in $1
do
for prune_ratio_per_epoch in 0.01 0.02 0.05 0.10
do
for ema in 0.1 0.2 0.4 0.5 0.8
do
for lr in 1e-4 1e-5
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
  ETTm1)
    thres=0.001;;
  *)
    thres=0.01;;
  esac
for pred_len in 96
do
  torchrun --nproc_per_node=4 src/tsfm/run.py --use_multi_gpu --finetune_epochs 1 \
    --root_path ./datasets/$dir/ --data_path $data_path --prune_transformer --stride $seq_len \
    --model $model_name --model_size $model_size --apply_aux_loss True \
    --model_id $model_size'_prune_raw_'$thres'_pr'$pr'_ema'$ema'_full_shot_balance_lr'$lr --learning_rate $lr \
    --pruned_model_id $model_size'_prune_raw_'$thres'_pr'$pr'_ema'$ema'_full_shot' \
    --seq_len $seq_len --prune_expert_threshold $thres --autoregressive --valid_autoregressive \
    --pred_len $pred_len --use_weight_decay 1 --clip_grad_norm 1.0 \
    --do_training 1 --wo_test --prune_ratio_per_epoch $pr --prune_ema $ema \
    --batch_size 64 >> ./logs/tsfm/$data'_'$model_name'_'$model_size'_L'$seq_len'_H'$pred_len'_prune_raw_'$thres'_pr'$pr'_ema'$ema'_balance_lr'$lr.log 2>&1
done
done
done
done
done