#!/bin/bash

for i in {1..5}
do
    python predict_tcr_pmhc_interact.py --config configs/config-paired-cdr123-pmhc-interact.yml \
        --input_data_path data/MEL8_A0201_peptides.csv \
        --model_location checkpoints/paired-cdr123-pmhc-interaction/paired-cdr123-pmhc-interaction-model-fold-${i}.pt \
        --log_dir results/interaction-MEL8-bg-cdr123-closest/Fold_${i}/
done
