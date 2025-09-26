if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/tsfm/" ]; then
    mkdir ./logs/tsfm/
fi
i=0
model_name=TimeMoE
model_size=base
seq_len=3072
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
    case $data in
    ETTh1)
      thres=0.1; pr=0.01; ema=0.8; lr=1e-4;;
    ETTh2)
      thres=0.01; pr=0.01; ema=0.2; lr=1e-5;;
    ETTm1)
      thres=0.001; pr=0.01; ema=0.2; lr=1e-4;;
    ETTm2)
      thres=0.01; pr=0.10; ema=0.6; lr=1e-4;;
    weather)
      thres=0.01; pr=0.02; ema=0.6; lr=1e-4;;
    esac
    CUDA_VISIBLE_DEVICES=$[i+0] python -u src/tsfm/run.py \
      --root_path ./datasets/$dir/ --data_path $data_path --stride $seq_len \
      --model $model_name --model_size $model_size --apply_aux_loss True \
      --model_id $model_size'_prune_raw_'$thres'_pr'$pr'_ema'$ema'_full_shot_balance_lr'$lr --learning_rate $lr \
      --pruned_model_id $model_size'_prune_raw_'$thres'_pr'$pr'_ema'$ema'_full_shot' \
      --seq_len $seq_len --prune_expert_threshold $thres --autoregressive --valid_autoregressive \
      --pred_len $pred_len --use_weight_decay 1 --clip_grad_norm 1.0 \
      --do_training 1 --reload True --prune_ratio_per_epoch $pr --prune_ema $ema \
      --batch_size 512 >> ./logs/tsfm/$data'_'$model_name'_'$model_size'_L'$seq_len'_H'$pred_len'_prune_raw_'$thres'_pr'$pr'_ema'$ema'_balance_lr'$lr.log 2>&1 &
    i=$[i+1]
    [ $i -eq 8 ] && wait
    i=$[i%8]
  done
done
