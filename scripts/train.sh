#!/bin/bash

# pretrain epitope masked language model.
python pretrain_plm.py --config configs/config-pretrain-epitope-lm.yml

# pretrain paired cdr3 masked language model.
python pretrain_plm.py --config configs/config-pretrain-cdr3-lm.yml

# pretrain paired cdr123 masked language model.
python pretrain_plm.py --config configs/config-pretrain-cdr123-lm.yml

# pretrain peptide-MHC binding affinity model.
python pretrain_pmhc_model.py --config configs/config-pmhc-binding.yml

# pretrain peptide-MHC eluted ligand model.
python pretrain_pmhc_model.py --config configs/config-pmhc-elution.yml

# finetune Paired TCR-pMHC binding model (CDR3).
python train_tcr_pmhc_binding.py --config configs/config-paired-cdr3-pmhc-binding.yml 

# finetune Paired TCR-pMHC binding model (CDR123).
python train_tcr_pmhc_binding.py --config configs/config-paired-cdr123-pmhc-binding.yml

# finetune Paired TCR-pMHC interaction model (CDR123).
python train_tcr_pmhc_interact.py --config configs/config-paired-cdr123-pmhc-interact.yml
