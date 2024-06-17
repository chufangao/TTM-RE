#!/bin/sh

python train2.py --data_dir ./dataset/docred \
    --transformer_type roberta \
    --model_name_or_path microsoft/deberta-v3-large \
    --train_file train_revised.json \
    --dev_file dev_revised.json \
    --test_file test_revised.json \
    --train_batch_size 4 \
    --test_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --num_train_epochs 30.0 \
    --seed 70 \
    --num_class 97 \
    --isrank 1 \
    --m_tag S-PU \
    --model_type ttmre \
    --m 1.0 \
    --e 1.0 \
    --pretrain_distant 2\
    --num_layers 2