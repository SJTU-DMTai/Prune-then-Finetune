if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi

model_name=TTM
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
for seq_len in 1536
do
for pred_len in 96 192 336 720
do
  filename=./logs/tsfm/$data'_'$model_name'a_L'$seq_len'_H'$pred_len'_lr{}.log'
  read lr <<< "$(python src/tsfm/select_hp.py --filename $filename --learning_rate 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9)"
  python -u src/tsfm/run.py \
    --root_path ./datasets/$dir/ --data_path $data_path --finetune_epochs 10 \
    --model $model_name \
    --model_id full_shot_lr$lr \
    --seq_len $seq_len --mode M \
    --pred_len $pred_len \
    --do_training 1 --learning_rate $lr \
    --batch_size 64 >> ./logs/tsfm/$data'_'$model_name'a_L'$seq_len'_H'$pred_len'_lr'$lr.log 2>&1
done
done
done