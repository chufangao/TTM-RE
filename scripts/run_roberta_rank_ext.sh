#!/bin/sh

python train2.py --data_dir ./dataset/docred \
    --transformer_type roberta \
    --model_name_or_path roberta-large \
    --train_file train_ext.json \
    --dev_file dev_ext.json \
    --test_file test_revised.json \
    --train_batch_size 8 \
    --test_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --num_train_epochs 30 \
    --seed 67 \
    --num_class 97 \
    --isrank 1 \
    --m_tag ATLoss \
    --model_type ATLOP \
    --m 1.0 \
    --e 3.0 \
    --pretrain_distant 0