#!/bin/sh

python train2.py --data_dir ./dataset/chemdisgene \
    --transformer_type bert \
    --model_name_or_path microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract \
    --train_file train.json \
    --dev_file valid.json \
    --test_file test.anno_all.json \
    --train_batch_size 8 \
    --test_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --num_train_epochs 30.0 \
    --seed 0 \
    --num_class 15 \
    --isrank 1 \
    --m_tag S-PU \
    --model_type ttmre \
    --m 0.25 \
    --e 3.0 \
    --pretrain_distant 0

# cd ChemDisGene
# python tsv2json.py ../../data ../../data
# test risk: 0.535857101460476 {'test_F1': 53.5857101460476, 're_p': 53.829675912857056, 're_r': 53.34395170211792}