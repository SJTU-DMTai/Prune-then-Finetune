if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi
i=0

model_name=moirai
model_size=base
mode=S
for data in $1
do
for lr in 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9
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
    ETTm2)
    P=128;;
    *)
    P=64;;
  esac
  case $data in
    ETTh2|ETTm2)
    bs=64
    mode=M;;
    *)
    bs=512
    mode=S;;
  esac
  case $data in
    ETTh2|ETTm2)
    L=4096;;
    ETTh1|ETTm1)
    L=2048;;
    *)
    L=3072;;
  esac
for pred_len in 96 192 336 720
do
  torchrun --nproc_per_node=4 src/tsfm/run.py --use_multi_gpu --finetune_epochs 1 \
    --root_path ./datasets/$dir/ --data_path $data_path \
    --model $model_name --model_size $model_size \
    --model_id $model_size'_'$mode'_full_shot_lr'$lr \
    --seq_len $L --patch_len $P --mode $mode \
    --pred_len $pred_len \
    --do_training 1 --wo_test --reload False --learning_rate $lr \
    --batch_size $bs >> ./logs/tsfm/$data'_'$model_name'_'$model_size'_'$mode'_L'$L'_H'$pred_len'_P'$P'_lr'$lr.log 2>&1
done
done
done