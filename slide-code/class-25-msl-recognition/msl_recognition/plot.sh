#!/bin/bash

python src/plot_results.py \
    --exp results/exp_bilstm results/exp_transformer results/exp_stgcn \
    --output results/model_comparison.png
