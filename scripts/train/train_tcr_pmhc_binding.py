from argparse import ArgumentParser
import os

from torch.utils.data import DataLoader, Subset

from EPACT.utils import load_config, set_seed
from EPACT.utils.sampling import get_epitope_idx_from_fasta
from EPACT.dataset.data import PairedTCRpMHCDataset
from EPACT.dataset.batch_converter import PairedCDR3pMHCBatchConverter, PairedCDR123pMHCBatchConverter
from EPACT.trainer import PairedCDR3pMHCCoembeddingTrainer, PairedCDR123pMHCCoembeddingTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    
    pos_dataset = PairedTCRpMHCDataset(data_path = config.data.train_pos_data_path,
                                    hla_lib_path = config.data.hla_lib_path,
                                    mhc_pseudo_pos = pseudo_seq_pos,
                                    use_cdr123 = config.data.use_cdr123)
    
    train_set, val_set = [], []
    for i in range(5):
        train_idx = get_epitope_idx_from_fasta(pos_dataset, f'{config.data.kfold_data_path}/train_epitope_fold_{i+1}.fasta')
        val_subset = PairedTCRpMHCDataset(data_path = f'{config.data.kfold_data_path}/val_fold_{i+1}.csv', # use sampled dataset for validation
                                    hla_lib_path = config.data.hla_lib_path,
                                    mhc_pseudo_pos = pseudo_seq_pos,
                                    use_cdr123 = config.data.use_cdr123)
        train_set.append(Subset(pos_dataset, train_idx))
        val_set.append(val_subset)
    
    test_dataset = PairedTCRpMHCDataset(data_path = config.data.test_data_path,
                                hla_lib_path = config.data.hla_lib_path,
                                mhc_pseudo_pos = pseudo_seq_pos,
                                use_cdr123 = config.data.use_cdr123)
    if config.data.use_cdr123:
        batch_converter = PairedCDR123pMHCBatchConverter(max_mhc_len = config.model.mhc_seq_len, sample_cdr3 = False)
    else:
        batch_converter = PairedCDR3pMHCBatchConverter(max_mhc_len = config.model.mhc_seq_len, sample_cdr3 = False)
        
    test_loader = DataLoader(
            dataset = test_dataset, batch_size = config.training.test_batch_size, num_workers = config.training.num_workers,
            collate_fn = batch_converter, shuffle = False
    )
    
    for i_split, (train_set_this, val_set_this) in enumerate(zip(train_set, val_set)):
        print('Split {}'.format(i_split), 'Train:', len(train_set_this) * (1 + config.training.non_binding_ratio), 'Val:', len(val_set_this))
        train_loader = DataLoader(
            dataset = train_set_this, batch_size = config.training.train_batch_size,
            num_workers = config.training.num_workers, shuffle = True,
            collate_fn = batch_converter
        )
        
        val_loader = DataLoader(
            dataset = val_set_this, batch_size = config.training.test_batch_size,
            num_workers = config.training.num_workers, shuffle = False,
            collate_fn = batch_converter
        )
    
        log_dir = os.path.join(config.training.log_dir, f'Fold_{i_split+1}')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        if config.data.use_cdr123:
            Trainer = PairedCDR123pMHCCoembeddingTrainer(config, log_dir=log_dir)
        else:
            Trainer = PairedCDR3pMHCCoembeddingTrainer(config, log_dir=log_dir)
            
        Trainer.fit(train_loader, val_loader)
        Trainer.test(test_loader, model_location=os.path.join(log_dir, 'checkpoint.pt'))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config-paired-cdr3-pmhc-new-binding.yml')
    args = parser.parse_args()
    
    main(args)
    