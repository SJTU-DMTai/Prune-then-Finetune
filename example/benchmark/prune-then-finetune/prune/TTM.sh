if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi

model_name=TTM
model_size=base
L=1536
for data in $1
do
for prune_ratio_per_epoch in 0.01 0.02 0.05
do
for ema in 0.1 0.2 0.4 0.5 0.8
do
for pred_len in 96 192 336 720
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
    ETTm1|ETTm2)
    E=10;;
    *)
    E=1;;
  esac
  case $data in
  traffic)
    mbs=512
    bs=32;;
  *)
    mbs=1024
    bs=64;;
  esac
  python -u src/tsfm/run.py \
    --root_path ./datasets/$dir/ --data_path $data_path --finetune_epochs $E \
    --model $model_name --model_size $model_size \
    --model_id 'a_M_prune_pr'$prune_ratio_per_epoch'_ema'$ema'_full_shot' \
    --seq_len $L --mode M \
    --pred_len $pred_len --macro_batch_size $mbs \
    --do_training 1 --wo_test --prune_ratio_per_epoch $prune_ratio_per_epoch --prune_ema $ema \
    --batch_size $bs >> ./logs/tsfm/$data'_'$model_name'a_M_L'$L'_H'$pred_len'_prune_pr'$prune_ratio_per_epoch'_ema'$ema.log 2>&1
done
done
done
done