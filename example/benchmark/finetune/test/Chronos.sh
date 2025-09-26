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
  filename=./logs/tsfm/$data'_'$model_name'_'$model_size'_L'$L'_H64_lr{}.log'
  read lr <<< "$(python src/tsfm/select_hp.py --filename $filename --learning_rate 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9)"
  filename=./logs/tsfm/$data'_'$model_name'_'$model_size'_L'$L'_H'$pred_len'_lr'$lr'.log'
  for pred_len in 96 192 336 720
  do
    python -u src/tsfm/run.py \
      --root_path ./datasets/$dir/ --data_path $data_path \
      --model $model_name --model_size $model_size --task_name forecast \
      --model_id $model_size'_'$mode'_full_shot_lr'$lr \
      --seq_len $L \
      --pred_len $pred_len \
      --do_training 1 --reload True --learning_rate $lr \
      --batch_size 2048 >> $filename 2>&1
  done
done