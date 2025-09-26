if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi
i=0

model_name=moirai
model_size=large
mode=S
L=5000
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
    ETTm1)
    P=128;;
    *)
    P=64;;
  esac
  case $data in
    ETTh2)
    bs=64
    mode=M;;
    *)
    bs=512
    mode=S;;
  esac
  case $data in
    ETTh1)
    L=2048;;
    ETTm2)
    L=4096;;
    *)
    L=3072;;
  esac
for pred_len in 96 192 336 720
do
  filename=./logs/tsfm/$data'_'$model_name'_'$model_size'_'$mode'_L'$L'_H'$pred_len'_P'$P'_lr{}'.log
  read lr <<< "$(python src/tsfm/select_hp.py --filename $filename --learning_rate 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9)"
  CUDA_VISIBLE_DEVICES=$[i+0] python -u src/tsfm/run.py \
    --root_path ./datasets/$dir/ --data_path $data_path \
    --model $model_name --model_size $model_size --task_name forecast \
    --model_id $model_size'_'$mode'_full_shot_lr'$lr \
    --seq_len $L --patch_len $P --mode $mode \
    --pred_len $pred_len \
    --do_training 0 --reload True --learning_rate $lr \
    --batch_size 2048 >> $filename 2>&1 &
  i=$[i+1]
  [ $i -eq 8 ] && wait
  i=$[i%8]
done
done