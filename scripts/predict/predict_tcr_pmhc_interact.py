from argparse import ArgumentParser
import os

from torch.utils.data import DataLoader

from EPACT.utils import load_config, set_seed
from EPACT.dataset import UnlabeledDataset, UnlabeledBacthConverter
from EPACT.trainer import PairedCDR3pMHCInteractTrainer, PairedCDR123pMHCInteractTrainer


def main(args):
    # load config
    config_path = args.config
    config = load_config(config_path)
    
    set_seed(config.training.seed)
    
    if config.model.mhc_seq_len == 34:
        pseudo_seq_pos = [7, 9, 24, 45, 59, 62, 63, 66, 67, 79, 70, 73, 74, 76, 77, 80, 81, 84, 95, 97,
                          99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171]  # NetMHCpan pseudosequence
    elif config.model.mhc_seq_len == 30:
        pseudo_seq_pos = [2, 9, 10, 33, 35, 48, 86, 90, 91, 93, 94, 95, 100, 101, 105, 119, 121,
                          138, 140, 176, 180, 182, 187, 206, 316, 317, 319, 320, 329, 339]  # BigMHC pseudosequence
    else:
        pseudo_seq_pos = None
        
    dataset = UnlabeledDataset(data_path = args.input_data_path,
                                hla_lib_path = config.data.hla_lib_path,
                                mhc_pseudo_pos = pseudo_seq_pos)
    
    data_loader = DataLoader(
        dataset = dataset, batch_size = config.training.batch_size, num_workers = config.training.num_workers,
        collate_fn = UnlabeledBacthConverter(max_mhc_len = config.model.mhc_seq_len, use_cdr123=config.data.use_cdr123),
        shuffle = False
    )
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    if config.data.use_cdr123:
        Trainer = PairedCDR123pMHCInteractTrainer(config, args.log_dir)
    else:
        Trainer = PairedCDR3pMHCInteractTrainer(config, args.log_dir)
    Trainer.predict(data_loader, model_location=args.model_location)
    
    
if __name__ == '__main__':
    
    '''
    python predict_tcr_pmhc_interact.py --config configs/config-paired-cdr3-pmhc-interact.yml \
                                        --input_data_path <input_data_path> \
                                        --model_location <checkpoint_path> \
                                        --log_dir <log_dir>
    '''
    
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config-interact.yml')
    parser.add_argument('--input_data_path', type=str, required=True)
    parser.add_argument('--model_location', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default='./')
    args = parser.parse_args()
    
    main(args)
    