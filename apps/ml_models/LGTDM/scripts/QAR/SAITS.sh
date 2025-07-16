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
      --model SAITS \
      --dataset QAR \
      --dataset_type QARDataset \
      --missing_ratio $missing_ratio \
      --scale 1 \
      --inverse_transform 1 \
      --split_len 3000 \
      --seq_len 150 \
      --saits_diagonal_attention_mask 1 \
      --saits_n_groups 2 \
      --saits_n_group_inner_layers 1 \
      --saits_MIT 1 \
      --saits_d_model 256 \
      --saits_d_inner 128 \
      --saits_n_head 4 \
      --saits_d_k 64 \
      --saits_d_v 64 \
      --saits_dropout 0.1 \
      --saits_input_with_mask 1 \
      --saits_param_sharing_strategy inner_group \
      --saits_imputation_loss_weight 1 \
      --saits_reconstruction_loss_weight 1 \
      --train_epochs 200 \
      --test_epoch_start 150 \
      --loss l1 \
      --learning_rate 0.001 \
      --batch_size 128 \
      --gpu 0
elif [[ "$missing_ratio" == "0.2" ]]; then
    python -u main.py \
      --is_training 1 \
      --model SAITS \
      --dataset QAR \
      --dataset_type QARDataset \
      --missing_ratio $missing_ratio \
      --scale 1 \
      --inverse_transform 1 \
      --split_len 3000 \
      --seq_len 150 \
      --saits_diagonal_attention_mask 1 \
      --saits_n_groups 2 \
      --saits_n_group_inner_layers 1 \
      --saits_MIT 1 \
      --saits_d_model 256 \
      --saits_d_inner 128 \
      --saits_n_head 4 \
      --saits_d_k 64 \
      --saits_d_v 64 \
      --saits_dropout 0.1 \
      --saits_input_with_mask 1 \
      --saits_param_sharing_strategy inner_group \
      --saits_imputation_loss_weight 1 \
      --saits_reconstruction_loss_weight 1 \
      --train_epochs 200 \
      --test_epoch_start 150 \
      --loss l1 \
      --learning_rate 0.001 \
      --batch_size 128 \
      --gpu 1
elif [[ "$missing_ratio" == "0.5" ]]; then
    python -u main.py \
      --is_training 1 \
      --model SAITS \
      --dataset QAR \
      --dataset_type QARDataset \
      --missing_ratio $missing_ratio \
      --scale 1 \
      --inverse_transform 1 \
      --split_len 3000 \
      --seq_len 150 \
      --saits_diagonal_attention_mask 1 \
      --saits_n_groups 2 \
      --saits_n_group_inner_layers 1 \
      --saits_MIT 1 \
      --saits_d_model 256 \
      --saits_d_inner 128 \
      --saits_n_head 4 \
      --saits_d_k 64 \
      --saits_d_v 64 \
      --saits_dropout 0.1 \
      --saits_input_with_mask 1 \
      --saits_param_sharing_strategy inner_group \
      --saits_imputation_loss_weight 1 \
      --saits_reconstruction_loss_weight 1 \
      --train_epochs 200 \
      --test_epoch_start 150 \
      --loss l1 \
      --learning_rate 0.001 \
      --batch_size 128 \
      --gpu 2
elif [[ "$missing_ratio" == "0.9" ]]; then
    python -u main.py \
      --is_training 1 \
      --model SAITS \
      --dataset QAR \
      --dataset_type QARDataset \
      --missing_ratio $missing_ratio \
      --scale 1 \
      --inverse_transform 1 \
      --split_len 3000 \
      --seq_len 150 \
      --saits_diagonal_attention_mask 1 \
      --saits_n_groups 2 \
      --saits_n_group_inner_layers 1 \
      --saits_MIT 1 \
      --saits_d_model 256 \
      --saits_d_inner 128 \
      --saits_n_head 4 \
      --saits_d_k 64 \
      --saits_d_v 64 \
      --saits_dropout 0.1 \
      --saits_input_with_mask 1 \
      --saits_param_sharing_strategy inner_group \
      --saits_imputation_loss_weight 1 \
      --saits_reconstruction_loss_weight 1 \
      --train_epochs 200 \
      --test_epoch_start 150 \
      --loss l1 \
      --learning_rate 0.001 \
      --batch_size 128 \
      --gpu 3
else
    echo "Unsupported missing_ratio: $missing_ratio"
    exit 1
fi