if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi
seq_len=4096
model_name=TimeMoE
model_size=base
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
for lr in 1e-4 1e-5
do
for pred_len in 96
do
  torchrun --nproc_per_node=4 src/tsfm/run.py --use_multi_gpu \
    --root_path ./datasets/$dir/ --data_path $data_path \
    --model $model_name --model_size $model_size --finetune_epochs 1 \
    --model_id $model_size'_full_shot_lr'$lr --use_weight_decay 1 --clip_grad_norm 1.0 \
    --seq_len $seq_len --stride $seq_len \
    --pred_len $pred_len \
    --do_training 1 --wo_test --learning_rate $lr --autoregressive --valid_autoregressive \
    --batch_size 32 >> ./logs/tsfm/$data'_'$model_name'_'$model_size'_L'$seq_len'_H'$pred_len'_lr'$lr.log 2>&1
done
done
done