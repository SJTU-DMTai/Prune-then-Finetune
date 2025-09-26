if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi
i=0

model_name=TimesFM
model_size=large
L=2048
for data in $1
do
for lr in 1e-7 1e-8 1e-9
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
    pr=0.05; ema=0.8;;
  ETTh2)
    pr=0.02; ema=0.8;;
  ETTm1)
    pr=0.15; ema=0.6;;
  ETTm2)
    pr=0.10; ema=0.6;;
  esac
for pred_len in 128
do
  filename=./logs/tsfm/$data'_'$model_name'_L'$L'_H'$pred_len'_pr'$pr'_ema'$ema'_lr'$lr.log
  torchrun --nproc_per_node=4 src/tsfm/run.py --use_multi_gpu --finetune_epochs 1 \
    --root_path ./datasets/$dir/ --data_path $data_path \
    --model $model_name --model_size $model_size --prune_transformer \
    --model_id 'pr'$pr'_ema'$ema'_full_shot_lr'$lr --learning_rate $lr \
    --pruned_model_id 'pr'$pr'_ema'$ema'_full_shot' \
    --seq_len $L \
    --pred_len $pred_len \
    --do_training 1 --wo_test --prune_ratio_per_epoch $pr --prune_ema $ema \
    --batch_size 128 >> $filename 2>&1
done
done
done