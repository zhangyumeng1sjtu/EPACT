#!/bin/bash

for i in {1..5}
do
    python predict_tcr_pmhc_binding.py \
        --config configs/config-paired-cdr123-pmhc-binding.yml \
        --input_data_path data/binding/Full-TCR/k-fold-data/val_fold_${i}.csv \
        --model_location checkpoints/paired-cdr123-pmhc-binding/paired-cdr123-pmhc-binding-model-fold-${i}.pt\
        --log_dir results/preds-cdr123-pmhc-binding/Fold_${i}/
done
