import argparse
from argparse import *

def parse_args():
    parser = ArgumentParser(description='''
    Written by Yongbin Kim, Nov 2025
    This is a wrapper program to run AI/ML ADME predictions

    Example usage:
    ''')

    # parser.add_argument('iFile',
    #                     type=str,
    #                     required=True,
    #                     help='dataset file used for train/predict/tuning in csv format')

    parser.add_argument('-p',
                        '--property',
                        type=str,
                        default='Caco2_Wang',
                        choices=['Caco2_Wang', 'bbb_martins', 'CYP3A4_Veith', 'Half_Life_Obach'],
                        help='specify the ADME property to predict,\
                        Caco2_Wang (A)\
                        bbb_martins (D)/\
                        CYP3A4_Veith (C)/\
                        Half_Life_Obach (E)\
                        default=Caco2_Wang')

    parser.add_argument('-t',
                        '--task',
                        type=str,
                        default='train',
                        choices=['train', 'predict', 'tune'],
                        help='specify the task, train/predict/tune, default=train')

    parser.add_argument('-d',
                        '--descriptor',
                        type=str,
                        default='maccs',
                        choices=['maccs', 'mpnn'],
                        help='specify the descriptor, maccs/mpnn, default=maccs')
    
    parser.add_argument('--smiles_col',
                        type=str,
                        default='Drug',
                        help='specify the smiles column name, default=Drug')
    
    parser.add_argument('--target_col',
                        type=str,
                        default='Y',
                        help='specify the target column name, default=Y')

    parser.add_argument('--split_method',
                        type=str,
                        default='scaffold',
                        choices=['scaffold', 'random'],
                        help='specify the split method, scaffold/random, default=scaffold')

    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default='0.01',
                        help='specify the learning rate, default=0.01')

    parser.add_argument('--batch',
                        type=int,
                        default=16,
                        help='specify the batch size, default=16')

    parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        help='specify the epochs, default=5')

    parser.add_argument('-pd',
                        '--param_dir',
                        type=str,
                        help='path/to/stored_param for transfer learning')
    
    parser.add_argument('-data',
                        '--data_dir',
                        type=str,
                        help='path/to/data for prediction')

    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help='random seed (optional)')
    
    # PyTorch trainer hyperparameters
    parser.add_argument('--input-shape',
                        type=int,
                        default=167,
                        help='input feature dimension for MACCS model (default=167)')

    parser.add_argument('--hidden-units',
                        type=int,
                        nargs='+',              # multiple ints: e.g. --hidden-units 128 64 32
                        default=[128, 64, 32],
                        help='hidden layer sizes for MACCS FCN (e.g. --hidden-units 128 64 32)')

    parser.add_argument('--output-shape',
                        type=int,
                        default=1,
                        help='output dimension (default=1 for regression)')

    args = parser.parse_args()
    if args.task == 'tuning' and not (args.param_dir and args.data_dir):
        parser.error('argument -pd/--param_dir and -data/--data_dir are required when task is \'predict\' or \'tuning\'.')

    return args
