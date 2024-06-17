#!/bin/sh

python train2.py --data_dir ./dataset/docred \
    --transformer_type roberta \
    --model_name_or_path roberta-large \
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
    --seed 74 \
    --num_class 97 \
    --isrank 1 \
    --m_tag S-PU \
    --model_type ttmre \
    --m 1.0 \
    --e 1.0 \
    --pretrain_distant 2\
    --num_layers 4


# 3e-5, finetune {'test_F1': 83.6041966673524, 'test_F1_ign': 82.67186522190119, 're_p': 85.79528318957718, 're_r': 81.52223750573133}
# 2e-5, finetune {'test_F1': 83.54658078741986, 'test_F1_ign': 82.54340770647327, 're_p': 84.5995651918444, 're_r': 82.51948647409445}
