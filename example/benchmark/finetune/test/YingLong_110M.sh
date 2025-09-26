if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi
export HF_ENDPOINT="https://hf-mirror.com"
i=0
model_name=YingLong
model_size=110M
seq_len=4096
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
  for pred_len in 96 192 336 720
  do
    filename=./logs/tsfm/$data'_'$model_name'_'$model_size'_L'$seq_len'_H'$pred_len'_lr{}.log'
    read lr <<< "$(python src/tsfm/select_hp.py --filename $filename --learning_rate 1e-5 1e-6 1e-7)"
    CUDA_VISIBLE_DEVICES=$[i+0] python -u src/tsfm/run.py --finetune_epochs 1 \
      --root_path ./datasets/$dir/ --data_path $data_path \
      --model $model_name --model_size $model_size --task_name forecast \
      --model_id $model_size'_full_shot_lr'$lr --input_ensemble \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --do_training 1 --reload True --learning_rate $lr \
      --batch_size 128 >> ./logs/tsfm/$data'_'$model_name'_'$model_size'_L'$seq_len'_H'$pred_len'_lr'$lr.log 2>&1 &
    i=$[i+1]
    [ $i -eq 8 ] && wait
    i=$[i%8]
  done
done
