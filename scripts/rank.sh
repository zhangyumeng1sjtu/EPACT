#!/bin/bash

python predict_tcr_pmhc_binding_rank.py --config configs/config-paired-cdr123-pmhc-binding.yml \
                                        --log_dir results/ranking-covid-cdr123/ \
                                        --input_data_path data/binding/covid_clonotypes.csv \
                                        --model_location checkpoints/paired-cdr123-pmhc-binding/paired-cdr123-pmhc-binding-model-all.pt \
                                        --bg_tcr_path data/pretrained/10x-paired-healthy-human-tcr-repertoire.csv \
                                        --num_bg_tcrs 20000