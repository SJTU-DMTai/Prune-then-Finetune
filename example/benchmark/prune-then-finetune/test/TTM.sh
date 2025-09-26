if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi
i=0

model_name=TTM
model_size=base
L=1536
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
for pred_len in 96 192 336 720
do
  filename=./logs/tsfm/$data'_'$model_name'a_M_L'$L'_H'$pred_len'_prune_pr{}_ema{}'.log
  read pr ema <<< "$(python src/tsfm/select_hp.py --prune_ratio_per_epoch 0.01 0.02 0.05 --filename $filename --prune_ema 0.1 0.2 0.4 0.5 0.8)"
  filename=./logs/tsfm/$data'_'$model_name'a_M_L'$L'_H'$pred_len'_prune_pr{}_ema{}_lr{}'.log
  read pr ema lr <<< "$(python src/tsfm/select_hp.py --prune_ratio_per_epoch $pr --filename $filename --prune_ema $ema --learning_rate 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9)"
#  filename=./logs/tsfm/$data'_'$model_name'a_M_L'$L'_H'$pred_len'_prune_pr{}_ema{}_lr{}'.log
#  read pr ema lr <<< "$(python src/tsfm/select_hp.py --prune_ratio_per_epoch 0.01 0.02 0.05 --filename $filename --prune_ema 0.1 0.2 0.4 0.5 0.8 --learning_rate 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9)"
  filename=./logs/tsfm/$data'_'$model_name'a_M_L'$L'_H'$pred_len'_bestpr_prune_pr'$pr'_ema'$ema'_lr'$lr.log
  echo $filename
  CUDA_VISIBLE_DEVICES=$[i+0] python -u src/tsfm/run.py --finetune_epoch 10 \
    --root_path ./datasets/$dir/ --data_path $data_path \
    --model $model_name --model_size $model_size \
    --pruned_model_id 'a_M_prune_pr'$pr'_ema'$ema'_full_shot' \
    --model_id 'a_M_prune_pr'$pr'_ema'$ema'_full_shot_lr'$lr --learning_rate $lr \
    --seq_len $L --mode M \
    --pred_len $pred_len --tag bestpr \
    --do_training 1 --reload True --prune_ratio_per_epoch $pr --prune_ema $ema \
    --batch_size 256 >> $filename 2>&1 &
  filename=./logs/tsfm/$data'_'$model_name'a_M_L'$L'_H'$pred_len'_prune_pr{}_ema{}_lr{}'.log
  read pr ema lr <<< "$(python src/tsfm/select_hp.py --prune_ratio_per_epoch 0.01 0.02 0.05 --filename $filename --prune_ema 0.1 0.2 0.4 0.5 0.8 --learning_rate 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9)"
  filename=./logs/tsfm/$data'_'$model_name'a_M_L'$L'_H'$pred_len'_best_pr_prune_pr'$pr'_ema'$ema'_lr'$lr.log
  echo $filename
  CUDA_VISIBLE_DEVICES=$[i+0] python -u src/tsfm/run.py --finetune_epoch 10 \
    --root_path ./datasets/$dir/ --data_path $data_path \
    --model $model_name --model_size $model_size \
    --pruned_model_id 'a_M_prune_pr'$pr'_ema'$ema'_full_shot' \
    --model_id 'a_M_prune_pr'$pr'_ema'$ema'_full_shot_lr'$lr --learning_rate $lr \
    --seq_len $L --mode M \
    --pred_len $pred_len --tag best \
    --do_training 1 --reload True --prune_ratio_per_epoch $pr --prune_ema $ema \
    --batch_size 256 >> $filename 2>&1 &
  i=$[i+1]
  [ $i -eq 4 ] && wait
  i=$[i%4]
done
done