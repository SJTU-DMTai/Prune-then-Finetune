if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi
i=0

model_name=TimesFM
model_size=base
L=2048
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
  ETTh1)
    pr=0.05; ema=0.8;;
  ETTh2)
    pr=0.02; ema=0.8;;
  ETTm1)
    pr=0.15; ema=0.6;;
  ETTm2)
    pr=0.10; ema=0.6;;
  esac
for pred_len in 96 192 336 720
do
  filename=./logs/tsfm/$data'_'$model_name'_L'$L'_H'$pred_len'_prune_pr'$pr'_ema'$ema'_lr'$lr.log
  CUDA_VISIBLE_DEVICES=$[i+0] python -u src/tsfm/run.py \
    --root_path ./datasets/$dir/ --data_path $data_path \
    --model $model_name --model_size $model_size --prune_transformer \
    --pruned_model_id 'prune_pr'$pr'_ema'$ema'_full_shot' --pruned_pred_len 128 --patch_len 128 \
    --model_id 'prune_pr'$pr'_ema'$ema'_full_shot_lr'$lr --learning_rate $lr \
    --seq_len $L \
    --pred_len $pred_len \
    --do_training 1 --reload True --prune_ratio_per_epoch $pr --prune_ema $ema \
    --batch_size 1024 >> $filename 2>&1 &
  i=$[i+1]
  [ $i -eq 8 ] && wait
  i=$[i%8]
done
done