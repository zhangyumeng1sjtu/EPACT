from argparse import ArgumentParser
import os
import pickle

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold

from EPACT.dataset import PairedTCRpMHCInteractDataset, PairedCDR3pMHCInteractBatchConverter, PairedCDR123pMHCInteractBatchConverter
from EPACT.trainer import PairedCDR3pMHCInteractTrainer, PairedCDR123pMHCInteractTrainer
from EPACT.utils import load_config, set_seed


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
        pseudo_seq_pos = None  # whole MHC sequence mhc_seq_len=366
    
    dataset = PairedTCRpMHCInteractDataset(
            data_path = config.data.train_data_path,
            pickle_path = config.data.pickle_path,
            hla_lib_path = config.data.hla_lib_path,
            mhc_pseudo_pos = pseudo_seq_pos,
            use_cdr123 = config.data.use_cdr123
        )
    if config.data.use_cdr123:
        batch_converter = PairedCDR123pMHCInteractBatchConverter(max_mhc_len=config.model.mhc_seq_len)
    else:
        batch_converter = PairedCDR3pMHCInteractBatchConverter(max_mhc_len=config.model.mhc_seq_len)
    
    with open(config.data.epi_cluster_path, 'rb') as f:
        epi2cluster = pickle.load(f)       
    cluster_id = [epi2cluster[data['epitope_seq']] for data in dataset]
    
    kfold = GroupKFold(5)
    splits = kfold.split(dataset, groups=cluster_id)
    
    train_set, val_set = [], []
    for train_idx, val_idx in splits:
        train_set.append(Subset(dataset, train_idx))
        val_set.append(Subset(dataset, val_idx)) 
    
    for i_split, (train_set_this, val_set_this) in enumerate(zip(train_set, val_set)):
        print('Split {}'.format(i_split), 'Train:', len(train_set_this), 'Val:', len(val_set_this))

        train_loader = DataLoader(train_set_this, batch_size=config.training.batch_size, collate_fn=batch_converter,
            num_workers=config.training.num_workers, shuffle=True)
        val_loader = DataLoader(val_set_this, batch_size=config.training.batch_size, collate_fn=batch_converter,
            num_workers=config.training.num_workers, shuffle=False)

        log_dir = os.path.join(config.training.log_dir, f'Fold_{i_split+1}')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        if config.data.use_cdr123:
            Trainer = PairedCDR123pMHCInteractTrainer(config, log_dir=log_dir)
        else:
            Trainer = PairedCDR3pMHCInteractTrainer(config, log_dir=log_dir)
            
        Trainer.fit(train_loader, val_loader)
        Trainer.test(val_loader, model_location=os.path.join(log_dir, 'checkpoint.pt'))
        
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config-paired-cdr123-pmhc-interact.yml')
    args = parser.parse_args()
    
    main(args)
