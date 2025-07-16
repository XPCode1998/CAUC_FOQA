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
      --model BRITS \
      --dataset QAR \
      --dataset_type QARDataset \
      --missing_ratio $missing_ratio \
      --scale 1 \
      --inverse_transform 1 \
      --split_len 3000 \
      --seq_len 150 \
      --brits_rnn_hid_size 64 \
      --train_epochs 200 \
      --test_epoch_start 150 \
      --loss l1 \
      --learning_rate 0.001 \
      --batch_size 32 \
      --gpu 0
elif [[ "$missing_ratio" == "0.2" ]]; then
    python -u main.py \
      --is_training 1 \
      --model BRITS \
      --dataset QAR \
      --dataset_type QARDataset \
      --missing_ratio $missing_ratio \
      --scale 1 \
      --inverse_transform 1 \
      --split_len 3000 \
      --seq_len 150 \
      --brits_rnn_hid_size 64 \
      --train_epochs 200 \
      --test_epoch_start 150 \
      --loss l1 \
      --learning_rate 0.001 \
      --batch_size 32 \
      --gpu 1
elif [[ "$missing_ratio" == "0.5" ]]; then
    python -u main.py \
      --is_training 1 \
      --model BRITS \
      --dataset QAR \
      --dataset_type QARDataset \
      --missing_ratio $missing_ratio \
      --scale 1 \
      --inverse_transform 1 \
      --split_len 3000 \
      --seq_len 150 \
      --brits_rnn_hid_size 64 \
      --train_epochs 200 \
      --test_epoch_start 150 \
      --loss l1 \
      --learning_rate 0.001 \
      --batch_size 32 \
      --gpu 2
elif [[ "$missing_ratio" == "0.9" ]]; then
    python -u main.py \
      --is_training 1 \
      --model BRITS \
      --dataset QAR \
      --dataset_type QARDataset \
      --missing_ratio $missing_ratio \
      --scale 1 \
      --inverse_transform 1 \
      --split_len 3000 \
      --seq_len 150 \
      --brits_rnn_hid_size 64 \
      --train_epochs 200 \
      --test_epoch_start 150 \
      --loss l1 \
      --learning_rate 0.001 \
      --batch_size 32 \
      --gpu 3
else
    echo "Unsupported missing_ratio: $missing_ratio"
    exit 1
fi