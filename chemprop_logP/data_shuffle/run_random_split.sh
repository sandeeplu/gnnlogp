#!/bin/bash

# Activate the Conda environment

source ~/miniconda3/bin/activate chemprop


##run the command for chemprop code

nohup chemprop train --data-path /home/a4724/chemprop/my_pred/random_split/with_other_para/data.csv \
    --task-type regression \
    --smiles-columns smiles --target-columns exp  --batch-size 32 --epoch 200 --loss-function rmse \
    --output-dir train_data \
    --split-type RANDOM --split-sizes 0.8 0.1 0.1  > output.log 2>&1 & 
