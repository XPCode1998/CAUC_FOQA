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
      --model LGTDM \
      --dataset QAR \
      --dataset_type QARDataset \
      --missing_ratio $missing_ratio \
      --scale 1 \
      --inverse_transform 1 \
      --split_len 3000 \
      --seq_len 150 \
      --diff_steps 30 \
      --diff_layers 4 \
      --res_channels 64 \
      --skip_channels 64 \
      --beta_start 0.0001 \
      --beta_end 0.5 \
      --beta_schedule quad \
      --step_emb_dim 64 \
      --c_step 128 \
      --unconditional 0 \
      --pos_emb_dim 0 \
      --fea_emb_dim 0 \
      --date_emb_dim 128 \
      --label_emb_dim 128 \
      --is_label_emb 1 \
      --c_cond 128 \
      --n_heads 8 \
      --e_layers 1 \
      --d_model 128 \
      --d_ff 256 \
      --factor 3 \
      --dropout 0.1 \
      --activation gelu \
      --tau_kernel_size 21 \
      --tau_dilation 2 \
      --only_generate_missing 1 \
      --train_missing_ratio_fixed 0 \
      --n_samples 100 \
      --select_k 1 \
      --train_epochs 150 \
      --test_epoch_start 100 \
      --loss huber \
      --learning_rate 0.0005 \
      --batch_size 8 \
      --gan_loss_ratio 0.1 \
      --gpu 0
elif [[ "$missing_ratio" == "0.2" ]]; then
    python -u main.py \
      --is_training 1 \
      --model LGTDM \
      --dataset QAR \
      --dataset_type QARDataset \
      --missing_ratio $missing_ratio \
      --scale 1 \
      --inverse_transform 1 \
      --split_len 3000 \
      --seq_len 150 \
      --diff_steps 30 \
      --diff_layers 4 \
      --res_channels 64 \
      --skip_channels 64 \
      --beta_start 0.0001 \
      --beta_end 0.5 \
      --beta_schedule quad \
      --step_emb_dim 64 \
      --c_step 128 \
      --unconditional 0 \
      --pos_emb_dim 0 \
      --fea_emb_dim 0 \
      --date_emb_dim 128 \
      --label_emb_dim 128 \
      --is_label_emb 1 \
      --c_cond 128 \
      --n_heads 8 \
      --e_layers 1 \
      --d_model 128 \
      --d_ff 256 \
      --factor 3 \
      --dropout 0.1 \
      --activation gelu \
      --tau_kernel_size 21 \
      --tau_dilation 2 \
      --only_generate_missing 1 \
      --train_missing_ratio_fixed 0 \
      --n_samples 100 \
      --select_k 1 \
      --train_epochs 150 \
      --test_epoch_start 100 \
      --loss huber \
      --learning_rate 0.0005 \
      --batch_size 8 \
      --gan_loss_ratio 0.1 \
      --gpu 1
elif [[ "$missing_ratio" == "0.5" ]]; then
    python -u main.py \
      --is_training 1 \
      --model LGTDM \
      --dataset QAR \
      --dataset_type QARDataset \
      --missing_ratio $missing_ratio \
      --scale 1 \
      --inverse_transform 1 \
      --split_len 3000 \
      --seq_len 150 \
      --diff_steps 30 \
      --diff_layers 4 \
      --res_channels 64 \
      --skip_channels 64 \
      --beta_start 0.0001 \
      --beta_end 0.5 \
      --beta_schedule quad \
      --step_emb_dim 64 \
      --c_step 128 \
      --unconditional 0 \
      --pos_emb_dim 0 \
      --fea_emb_dim 0 \
      --date_emb_dim 128 \
      --label_emb_dim 128 \
      --is_label_emb 1 \
      --c_cond 128 \
      --n_heads 8 \
      --e_layers 1 \
      --d_model 128 \
      --d_ff 256 \
      --factor 3 \
      --dropout 0.1 \
      --activation gelu \
      --tau_kernel_size 21 \
      --tau_dilation 2 \
      --only_generate_missing 1 \
      --train_missing_ratio_fixed 0 \
      --n_samples 100 \
      --select_k 1 \
      --train_epochs 150 \
      --test_epoch_start 100 \
      --loss huber \
      --learning_rate 0.0005 \
      --batch_size 8 \
      --gan_loss_ratio 0.1 \
      --gpu 2
elif [[ "$missing_ratio" == "0.9" ]]; then
    python -u main.py \
      --is_training 1 \
      --model LGTDM \
      --dataset QAR \
      --dataset_type QARDataset \
      --missing_ratio $missing_ratio \
      --scale 1 \
      --inverse_transform 1 \
      --split_len 3000 \
      --seq_len 150 \
      --diff_steps 30 \
      --diff_layers 4 \
      --res_channels 64 \
      --skip_channels 64 \
      --beta_start 0.0001 \
      --beta_end 0.5 \
      --beta_schedule quad \
      --step_emb_dim 64 \
      --c_step 128 \
      --unconditional 0 \
      --pos_emb_dim 0 \
      --fea_emb_dim 0 \
      --date_emb_dim 128 \
      --label_emb_dim 128 \
      --is_label_emb 1 \
      --c_cond 128 \
      --n_heads 8 \
      --e_layers 1 \
      --d_model 128 \
      --d_ff 256 \
      --factor 3 \
      --dropout 0.1 \
      --activation gelu \
      --tau_kernel_size 21 \
      --tau_dilation 2 \
      --only_generate_missing 1 \
      --train_missing_ratio_fixed 0 \
      --n_samples 100 \
      --select_k 1 \
      --train_epochs 150 \
      --test_epoch_start 100 \
      --loss huber \
      --learning_rate 0.0005 \
      --batch_size 8 \
      --gan_loss_ratio 0.1 \
      --gpu 3
else
    echo "Unsupported missing_ratio: $missing_ratio"
    exit 1
fi