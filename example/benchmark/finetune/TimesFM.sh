if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi
i=0

model_name=TimesFM
model_size=large
mode=S
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
for pred_len in 128
do
  torchrun --nproc_per_node=4 src/tsfm/run.py --use_multi_gpu \
    --root_path ./datasets/$dir/ --data_path $data_path \
    --model $model_name --model_size $model_size \
    --model_id full_shot_lr$lr --patch_len 128 \
    --seq_len $L \
    --pred_len $pred_len \
    --do_training 1 --wo_test --learning_rate $lr \
    --batch_size 128 >> ./logs/tsfm/$data'_'$model_name'_L'$L'_H'$pred_len'_lr'$lr.log 2>&1
done
done
done