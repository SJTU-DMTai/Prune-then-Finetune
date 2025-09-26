if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi
i=0
model_name=TimerXL
seq_len=2880
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
  filename=./logs/tsfm/$data'_'$model_name'_L'$L'_H96_P'$P'_prune_pr{}_ema{}_lr{}'.log
  read pr ema lr <<< "$(python src/tsfm/select_hp.py --prune_ratio_per_epoch $pr --filename $filename --prune_ema 0.1 0.2 0.4 0.5 0.8 --learning_rate 1e-5 1e-6 1e-7 1e-8 1e-9)"
    CUDA_VISIBLE_DEVICES=$[i+0] python -u src/tsfm/run.py \
      --root_path ./datasets/$dir/ --data_path $data_path \
      --model $model_name \
      --model_id 'prune_pr'$pr'_ema'$ema'_full_shot_lr'$lr --learning_rate $lr \
      --pruned_model_id 'prune_pr'$pr'_ema'$ema'_full_shot' \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --do_training 1 --reload True --prune_ratio_per_epoch $pr --prune_ema $ema --autoregressive \
      --batch_size 2048 >> ./logs/tsfm/$data'_'$model_name'_L'$seq_len'_H'$pred_len'_prune_pr'$pr'_ema'$ema'_lr'$lr.log 2>&1 &
  i=$[i+1]
  [ $i -eq 8 ] && wait
  i=$[i%8]
  done
done
