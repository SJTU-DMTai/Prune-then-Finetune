if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi
i=0

model_name=Chronos
model_size=base
mode=S
L=2048
for data in $1
do
for lr in 1e-3 1e-4 1e-5 1e-6
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
  python -u src/tsfm/run.py --finetune_epochs 1 \
    --root_path ./datasets/$dir/ --data_path $data_path \
    --model $model_name --model_size $model_size \
    --model_id $model_size'_'$mode'_full_shot_lr'$lr \
    --seq_len $L \
    --pred_len $pred_len \
    --do_training 1 --wo_test --learning_rate $lr \
    --batch_size 256 >> ./logs/tsfm/$data'_'$model_name'_'$model_size'_L'$L'_H'$pred_len'_lr'$lr.log 2>&1
done
done
done