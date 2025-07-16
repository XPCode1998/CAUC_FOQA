missing_ratio = 0.1


while [[ "$#" -gt 0 ]]; do
    case $1 in
        --missing_ratio) missing_ratio="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


if [[ "$missing_ratio" == "0.1" ]]; then
    python -u main.py \
      --is_training 1 \
      --model CSDI \
      --dataset ADSB \
      --dataset_type ADSBDataset \
      --missing_ratio $missing_ratio \
      --scale 0 \
      --inverse_transform 0 \
      --seq_len 100 \
      --diff_steps 50 \
      --diff_layers 4 \
      --res_channels 64 \
      --skip_channels 64 \
      --beta_start 0.0001 \
      --beta_end 0.5 \
      --beta_schedule quad \
      --step_emb_dim 64 \
      --c_step 128 \
      --unconditional 0 \
      --pos_emb_dim 128 \
      --fea_emb_dim 16 \
      --n_heads 8 \
      --e_layers 1 \
      --d_ff 64 \
      --n_samples 100 \
      --only_generate_missing 1 \
      --train_missing_ratio_fixed 0 \
      --train_epochs 200 \
      --test_epoch_start 150 \
      --loss l2 \
      --learning_rate 0.001 \
      --batch_size 16 \
      --gpu 0
elif [[ "$missing_ratio" == "0.2" ]]; then
    python -u main.py \
      --is_training 1 \
      --model CSDI \
      --dataset ADSB \
      --dataset_type ADSBDataset \
      --missing_ratio $missing_ratio \
      --scale 0 \
      --inverse_transform 0 \
      --seq_len 100 \
      --diff_steps 50 \
      --diff_layers 4 \
      --res_channels 64 \
      --skip_channels 64 \
      --beta_start 0.0001 \
      --beta_end 0.5 \
      --beta_schedule quad \
      --step_emb_dim 64 \
      --c_step 128 \
      --unconditional 0 \
      --pos_emb_dim 128 \
      --fea_emb_dim 16 \
      --n_heads 8 \
      --e_layers 1 \
      --d_ff 64 \
      --n_samples 100 \
      --only_generate_missing 1 \
      --train_missing_ratio_fixed 0 \
      --train_epochs 200 \
      --test_epoch_start 150 \
      --loss l2 \
      --learning_rate 0.001 \
      --batch_size 16 \
      --gpu 1
elif [[ "$missing_ratio" == "0.5" ]]; then
    python -u main.py \
      --is_training 1 \
      --model CSDI \
      --dataset ADSB \
      --dataset_type ADSBDataset \
      --missing_ratio $missing_ratio \
      --scale 0 \
      --inverse_transform 0 \
      --seq_len 100 \
      --diff_steps 50 \
      --diff_layers 4 \
      --res_channels 64 \
      --skip_channels 64 \
      --beta_start 0.0001 \
      --beta_end 0.5 \
      --beta_schedule quad \
      --step_emb_dim 64 \
      --c_step 128 \
      --unconditional 0 \
      --pos_emb_dim 128 \
      --fea_emb_dim 16 \
      --n_heads 8 \
      --e_layers 1 \
      --d_ff 64 \
      --n_samples 100 \
      --only_generate_missing 1 \
      --train_missing_ratio_fixed 0 \
      --train_epochs 200 \
      --test_epoch_start 150 \
      --loss l2 \
      --learning_rate 0.001 \
      --batch_size 16 \
      --gpu 2
elif [[ "$missing_ratio" == "0.9" ]]; then
    python -u main.py \
      --is_training 1 \
      --model CSDI \
      --dataset ADSB \
      --dataset_type ADSBDataset \
      --missing_ratio $missing_ratio \
      --scale 0 \
      --inverse_transform 0 \
      --seq_len 100 \
      --diff_steps 50 \
      --diff_layers 4 \
      --res_channels 64 \
      --skip_channels 64 \
      --beta_start 0.0001 \
      --beta_end 0.5 \
      --beta_schedule quad \
      --step_emb_dim 64 \
      --c_step 128 \
      --unconditional 0 \
      --pos_emb_dim 128 \
      --fea_emb_dim 16 \
      --n_heads 8 \
      --e_layers 1 \
      --d_ff 64 \
      --n_samples 100 \
      --only_generate_missing 1 \
      --train_missing_ratio_fixed 0 \
      --train_epochs 200 \
      --test_epoch_start 150 \
      --loss l2 \
      --learning_rate 0.001 \
      --batch_size 16 \
      --gpu 3
else
    echo "Unsupported missing_ratio: $missing_ratio"
    exit 1
fi