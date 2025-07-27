#!/bin/bash

# Activate the Conda environment

source ~/miniconda3/bin/activate pytorch_env 

#source ~/miniconda3/bin/activate pytorch_env > output.log 2>&1

# Run the Python script in the background with nohup
#nohup python gnn_mod.py > output.log 2>&1 &
nohup python crossval.py > output.log 2>&1 &

echo "Script is running in the background. Check 'output.log' for the output."
