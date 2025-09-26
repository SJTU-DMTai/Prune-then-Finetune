if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi
i=0

model_name=Chronos
model_size=base
L=2048
for data in $1
do
for lr in 1e-2 1e-3 1e-4 1e-5 1e-6
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
for pred_len in 64
do
  filename=./logs/tsfm/$data'_'$model_name'_'$model_size'_L'$L'_H'$pred_len'_prune_pr{}_ema{}'.log
  read pr ema <<< "$(python src/tsfm/select_hp.py --filename $filename --prune_ratio_per_epoch 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --prune_ema 0.1 0.2 0.4 0.5 0.8)"
  filename=./logs/tsfm/$data'_'$model_name'_'$model_size'_L'$L'_H'$pred_len'_prune_pr'$pr'_ema'$ema'_lr'$lr.log
  echo $filename
#  torchrun --nproc_per_node=4 src/tsfm/run.py --use_multi_gpu --finetune_epochs 1 \
  (
  CUDA_VISIBLE_DEVICES=$i python -u src/tsfm/run.py \
    --root_path ./datasets/$dir/ --data_path $data_path \
    --model $model_name --model_size $model_size --prune_transformer --finetune_epochs 1 \
    --model_id $model_size'_prune_pr'$pr'_ema'$ema'_full_shot_lr'$lr --learning_rate $lr \
    --pruned_model_id $model_size'_prune_pr'$pr'_ema'$ema'_full_shot' \
    --seq_len $L \
    --pred_len $pred_len \
    --do_training 1 --wo_test --prune_ratio_per_epoch $pr --prune_ema $ema \
    --batch_size 256 >> $filename 2>&1
  )&
  i=$[i+1]
  [ $i -eq 5 ] && wait
  i=$[i%5]
done
done
done