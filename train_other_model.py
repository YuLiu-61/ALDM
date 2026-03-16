#%%
from math import gamma
import os
import argparse
from statistics import mode
import torch
from models.DeepSAD import DeepSVDD,DeepSAD
from models.DROCC import DROCCTrainer, LSTM_FC #DROCC
from models.GAN import R_Net, D_Net, CNNAE, train_model, R_Loss, D_Loss, test_single_epoch
import numpy as np
from utils import ROC

# from data import fetch_dataloaders


parser = argparse.ArgumentParser()
# files
parser.add_argument('--data_dir', type=str, 
                    default='Data/input/SWaT_Dataset_Attack_v0.csv', help='Location of datasets.')
parser.add_argument('--output_dir', type=str, 
                    default='./checkpoint/model')
parser.add_argument('--name',default='PSM')
# restore
parser.add_argument('--graph', type=str, default='DeepSAD')
parser.add_argument('--model', type=str, choices = ['DeepSVDD', 'DeepSAD', 'DROCC', 'EncDecAD', 'ALOCC'] ,default='DeepSAD')
parser.add_argument('--seed', type=int, default=18, help='Random seed to use.')
parser.add_argument('--load', type=str, default="./checkpoint/model/PSM/PSM_19.pt")

# made parameters
parser.add_argument('--n_blocks', type=int, default=2, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--batch_norm', type=bool, default=False)

# training params
parser.add_argument('--train_split', type=float, default=0.6)
parser.add_argument('--stride_size', type=int, default=10)
parser.add_argument('--window_size', type=int, default=60)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--n_epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--log_interval', type=int, default=1, help='How often to show loss statistics and save samples.')


args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")



import random
import numpy as np
import math
#%%



for seed in range(15,20):
 
    args.seed = seed
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        
    from Dataset import load_smd_smap_msl, loader_SWat, loader_PSM, loader_tepe

    if args.name == 'SWaT':
        train_loader, val_loader, test_loader, n_sensor = loader_SWat(args.data_dir, \
                                                                        args.batch_size, args.window_size, args.stride_size, args.train_split)


    elif args.name == 'SMAP' or args.name == 'MSL' or args.name.startswith('machine'):
        train_loader, val_loader, test_loader, n_sensor = load_smd_smap_msl(args.name, \
                                                                    args.batch_size, args.window_size, args.stride_size, args.train_split)

    elif args.name == 'PSM':
        train_loader, val_loader, test_loader, n_sensor = loader_PSM(args.name, \
                                                                    args.batch_size, args.window_size, args.stride_size, args.train_split)

    elif args.name == 'TEP':
        train_loader, val_loader, test_loader, n_sensor = loader_tepe(args.name, \
                                                                     args.batch_size, args.window_size,
                                                                     args.stride_size, args.train_split)
    print("Loading dataset")
    print(args.name)
    

    if args.model == 'DeepSVDD':
        model = DeepSVDD(n_sensor, args.hidden_size, device)

        if args.load:
            model.ae_net.encoder.load_state_dict(torch.load(args.load)['net_dict'])
            c = torch.load(args.load)['c']
        model.train(train_loader, test_loader, args, device)
        gt, pre = model.test(test_loader, c,1, device)
        ROC(args, gt, pre)
    
    elif args.model == 'DeepSAD':
        model = DeepSAD(n_sensor, args.hidden_size, device)

        if args.load:
            model.ae_net.encoder.load_state_dict(torch.load(args.load)['net_dict'])
            c = torch.load(args.load)['c']
        model.train(train_loader, test_loader, args, device)
        gt, pre = model.test(test_loader, c,1, device)
        ROC(args, gt, pre)
        

    
