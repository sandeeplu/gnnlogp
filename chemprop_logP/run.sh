#!/bin/bash

# Activate the Conda environment

source ~/miniconda3/bin/activate chemprop


##run the command for chemprop code

nohup chemprop train --data-path /home/a4724/chemprop/my_pred/data.csv \
    --task-type regression \
    --smiles-columns smiles --target-columns exp  --batch-size 32 --epoch 200 --loss-function rmse \
    --metrics r2 --output-dir train_data \
    --split-type SCAFFOLD_BALANCED > output.log 2>&1 & 
