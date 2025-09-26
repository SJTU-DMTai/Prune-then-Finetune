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
for prune_ratio_per_epoch in 0.05 0.10 0.15 0.20
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
    ETTh2|ETTm2|M-dense)
    L=4096;;
    ETTh1|ETTm1)
    L=2048;;
    *)
    L=3072;;
  esac
for pred_len in 96 192 336 720
do
  torchrun --nproc_per_node=4 src/tsfm/run.py --use_multi_gpu \
    --root_path ./datasets/$dir/ --data_path $data_path \
    --model $model_name --model_size $model_size --prune_transformer \
    --model_id $model_size'_'$mode'_prune_pr'$prune_ratio_per_epoch'_ema'$ema'_full_shot' \
    --seq_len $L --patch_len $P --mode $mode \
    --pred_len $pred_len --macro_batch_size 8192 \
    --do_training 1 --wo_test --prune_ratio_per_epoch $prune_ratio_per_epoch --prune_ema $ema \
    --batch_size $bs >> ./logs/tsfm/$data'_'$model_name'_'$model_size'_'$mode'_L'$L'_H'$pred_len'_P'$P'_prune_pr'$prune_ratio_per_epoch'_ema'$ema.log 2>&1
done
done
done
done