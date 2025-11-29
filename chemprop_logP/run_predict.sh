#!/bin/bash

# Activate the Conda environment

source ~/miniconda3/bin/activate chemprop


##run the command for chemprop code

nohup chemprop predict \
  --test-path /home/a4724/chemprop/my_pred/data.csv \
  --model-path /home/a4724/chemprop/my_pred/train_data/model_0/best.pt \
  --preds-path predictions.csv \
  --smiles-columns smiles >output_pred.log 2>&1 &
