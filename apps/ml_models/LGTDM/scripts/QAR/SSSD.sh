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
      --model SSSD \
      --dataset QAR \
      --dataset_type QARDataset \
      --missing_ratio $missing_ratio \
      --scale 1 \
      --inverse_transform 1 \
      --split_len 3000 \
      --seq_len 150 \
      --diff_steps 200 \
      --diff_layers 36 \
      --res_channels 256 \
      --skip_channels 256 \
      --beta_start 0.0001 \
      --beta_end 0.02 \
      --beta_schedule quad \
      --step_emb_dim 128 \
      --c_step 512 \
      --s4_lmax 100 \
      --s4_d_state 64 \
      --s4_dropout 0.0 \
      --s4_bidirectional 1 \
      --s4_layernorm 1 \
      --only_generate_missing 1 \
      --train_missing_ratio_fixed 0 \
      --train_epochs 100 \
      --test_epoch_start 50 \
      --loss l2 \
      --learning_rate 0.0002 \
      --batch_size 16 \
      --gpu 0
elif [[ "$missing_ratio" == "0.2" ]]; then
    python -u main.py \
      --is_training 1 \
      --model SSSD \
      --dataset QAR \
      --dataset_type QARDataset \
      --missing_ratio $missing_ratio \
      --scale 1 \
      --inverse_transform 1 \
      --split_len 3000 \
      --seq_len 150 \
      --diff_steps 200 \
      --diff_layers 36 \
      --res_channels 256 \
      --skip_channels 256 \
      --beta_start 0.0001 \
      --beta_end 0.02 \
      --beta_schedule quad \
      --step_emb_dim 128 \
      --c_step 512 \
      --s4_lmax 100 \
      --s4_d_state 64 \
      --s4_dropout 0.0 \
      --s4_bidirectional 1 \
      --s4_layernorm 1 \
      --only_generate_missing 1 \
      --train_missing_ratio_fixed 0 \
      --train_epochs 100 \
      --test_epoch_start 50 \
      --loss l2 \
      --learning_rate 0.0002 \
      --batch_size 16 \
      --gpu 1
elif [[ "$missing_ratio" == "0.5" ]]; then
    python -u main.py \
      --is_training 1 \
      --model SSSD \
      --dataset QAR \
      --dataset_type QARDataset \
      --missing_ratio $missing_ratio \
      --scale 1 \
      --inverse_transform 1 \
      --split_len 3000 \
      --seq_len 150 \
      --diff_steps 200 \
      --diff_layers 36 \
      --res_channels 256 \
      --skip_channels 256 \
      --beta_start 0.0001 \
      --beta_end 0.02 \
      --beta_schedule quad \
      --step_emb_dim 128 \
      --c_step 512 \
      --s4_lmax 100 \
      --s4_d_state 64 \
      --s4_dropout 0.0 \
      --s4_bidirectional 1 \
      --s4_layernorm 1 \
      --only_generate_missing 1 \
      --train_missing_ratio_fixed 0 \
      --train_epochs 100 \
      --test_epoch_start 50 \
      --loss l2 \
      --learning_rate 0.0002 \
      --batch_size 16 \
      --gpu 2
elif [[ "$missing_ratio" == "0.9" ]]; then
    python -u main.py \
      --is_training 1 \
      --model SSSD \
      --dataset QAR \
      --dataset_type QARDataset \
      --missing_ratio $missing_ratio \
      --scale 1 \
      --inverse_transform 1 \
      --split_len 3000 \
      --seq_len 150 \
      --diff_steps 200 \
      --diff_layers 36 \
      --res_channels 256 \
      --skip_channels 256 \
      --beta_start 0.0001 \
      --beta_end 0.02 \
      --beta_schedule quad \
      --step_emb_dim 128 \
      --c_step 512 \
      --s4_lmax 100 \
      --s4_d_state 64 \
      --s4_dropout 0.0 \
      --s4_bidirectional 1 \
      --s4_layernorm 1 \
      --only_generate_missing 1 \
      --train_missing_ratio_fixed 0 \
      --train_epochs 100 \
      --test_epoch_start 50 \
      --loss l2 \
      --learning_rate 0.0002 \
      --batch_size 16 \
      --gpu 3
else
    echo "Unsupported missing_ratio: $missing_ratio"
    exit 1
fi