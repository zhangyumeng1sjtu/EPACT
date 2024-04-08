from argparse import ArgumentParser
import os
import pickle

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

from EPACT.dataset import pMHCDataset, pMHCBatchConverter
from EPACT.trainer import EpitopeMHCTrainer
from EPACT.utils import set_seed, load_config


def main(args):
    # load config
    config_path = args.config
    config = load_config(config_path)
    
    if not os.path.exists(config.training.log_dir):
        os.makedirs(config.training.log_dir)
    os.system(f'cp {args.config} {config.training.log_dir}/config.yml')
    
    set_seed(config.training.seed)
    
    if config.model.mhc_seq_len == 34:
        pseudo_seq_pos = [7, 9, 24, 45, 59, 62, 63, 66, 67, 79, 70, 73, 74, 76, 77, 80, 81, 84, 95, 97,
                          99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171]  # NetMHCpan pseudosequence
    elif config.model.mhc_seq_len == 30:
        pseudo_seq_pos = [2, 9, 10, 33, 35, 48, 86, 90, 91, 93, 94, 95, 100, 101, 105, 119, 121,
                          138, 140, 176, 180, 182, 187, 206, 316, 317, 319, 320, 329, 339]  # BigMHC pseudosequence
    else:
        pseudo_seq_pos = None
        
    if config.task == 'BA':
        with open(config.data.pep_cluster_path, 'rb') as f:
            pep2cluster = pickle.load(f)
        
        dataset = pMHCDataset(data_path = config.data.data_path,
                              hla_lib_path = config.data.hla_lib_path,
                              mhc_pseudo_pos = pseudo_seq_pos)
        
        rawdata = pd.read_csv(config.data.data_path)
        group_ids = [pep2cluster[pep] for pep in rawdata['Epitope.peptide']]
        
        train_test_split = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=config.training.seed)
        for train_idx, test_idx in train_test_split.split(rawdata, groups=group_ids):
            train_data = Subset(dataset, train_idx)
            test_data = Subset(dataset, test_idx)
        
        batch_converter = pMHCBatchConverter(max_epitope_len=config.data.max_epi_len, max_mhc_len=config.model.mhc_seq_len)
        test_loader = DataLoader(test_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                                 num_workers=config.training.num_workers, shuffle=False)
        
        train_loader = DataLoader(train_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                                      num_workers=config.training.num_workers, shuffle=True)
        Trainer = EpitopeMHCTrainer(config)
        Trainer.fit(train_loader, test_loader)
        Trainer.test(test_loader, model_location=os.path.join(config.training.log_dir, 'checkpoint.pt'))
            
    elif config.task == 'EL':
        train_data = pMHCDataset(data_path = os.path.join(config.data.data_path, 'el_train.csv'),
                              hla_lib_path = config.data.hla_lib_path, mhc_pseudo_pos = pseudo_seq_pos)
        val_data = pMHCDataset(data_path = os.path.join(config.data.data_path, 'el_val.csv'),
                              hla_lib_path = config.data.hla_lib_path, mhc_pseudo_pos = pseudo_seq_pos)
        test_data = pMHCDataset(data_path = os.path.join(config.data.data_path, 'el_test.csv'),
                              hla_lib_path = config.data.hla_lib_path, mhc_pseudo_pos = pseudo_seq_pos)
        
        print(f'Train data size: {len(train_data)}')
        print(f'Val data size: {len(val_data)}')
        print(f'Test data size: {len(test_data)}')
        
        batch_converter = pMHCBatchConverter(max_epitope_len=config.data.max_epi_len, max_mhc_len=config.model.mhc_seq_len)  
        train_loader = DataLoader(train_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                                      num_workers=config.training.num_workers, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                                      num_workers=config.training.num_workers, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                                      num_workers=config.training.num_workers, shuffle=False)

        Trainer =  EpitopeMHCTrainer(config)
        Trainer.fit(train_loader, val_loader)
        Trainer.test(test_loader, model_location=os.path.join(config.training.log_dir, 'checkpoint.pt'))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config-pmhc-binding.yml')
    args = parser.parse_args()
    
    main(args)
    