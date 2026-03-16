import os
import argparse
import torch
from models.ALDM import ALDM
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import auc


from utils import (
    compute_event_wise_metrics,
)

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str,
                    default='./Dataset/Data/input/SWaT_Dataset_Attack_v0.csv', help='Location of datasets.')
parser.add_argument('--output_dir', type=str,
                    default='./checkpoint/')
parser.add_argument('--name',default='SWaT', help='the name of dataset')

parser.add_argument('--graph', type=str, default='None')
parser.add_argument('--model', type=str, default='ALDM')


parser.add_argument('--n_blocks', type=int, default=1, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--dim_ml', type=int, default=128, help='Dimension of metric Adaptive Distance Computation')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--input_size', type=int, default=1)
parser.add_argument('--batch_norm', type=bool, default=False)
parser.add_argument('--train_split', type=float, default=0.8)
parser.add_argument('--stride_size', type=int, default=10)
parser.add_argument('--outdim_scl', type=int, default=320,help='Dimension of output layer for latent encoding.')
parser.add_argument('--hiddendim_scl', type=int, default=128, help='Dimension of hidden layers for latent encoding.')
parser.add_argument('--soft_instance', type=bool, default=False)
parser.add_argument('--soft_temporal', type=bool, default=True)

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--window_size', type=int, default=60)




args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
save_path = os.path.join(args.output_dir,args.name)

from Dataset import load_smd_smap_msl, loader_SWat,  loader_PSM, loader_tepe

if args.name == 'SWaT':
    train_loader, val_loader, test_loader, n_sensor = loader_SWat(args.data_dir, \
                                                                    args.batch_size, args.window_size, args.stride_size, args.train_split)


elif args.name == 'SMAP' or args.name == 'MSL' or args.name.startswith('machine'):
    train_loader, val_loader, test_loader, n_sensor = load_smd_smap_msl(args.name, \
                                                                args.batch_size, args.window_size, args.stride_size, args.train_split)

elif args.name == 'PSM':
    train_loader, val_loader, test_loader, n_sensor = loader_PSM(args.name, \
                                                                args.batch_size, args.window_size, args.stride_size, args.train_split)

elif args.name == 'TEPE':
    train_loader, val_loader, test_loader, n_sensor = loader_tepe(args.name, \
                                                                    args.batch_size, args.window_size, args.stride_size, args.train_split)



#%%
model = ALDM(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.window_size, n_sensor,
              args.outdim_scl, args.hiddendim_scl, args.soft_instance, args.soft_temporal, args.dim_ml, dropout=0.0,
              model=args.model, batch_norm=args.batch_norm)
model = model.to(device)

for seed in range(15,16):
    checkpoint = torch.load(f"./checkpoint/{args.name}/model_seed_{seed}.pth")
    model.load_state_dict(checkpoint['model'])


    model.eval()

    loss_test = []
    preds = []

    with torch.no_grad():
        for x, y, t in test_loader:
            x = x.to(device)
            loss, _,_= model.test(x, )
            loss = loss.cpu().numpy()
            loss_test.append(-loss)
        loss_test = np.concatenate(loss_test)
        y_true = np.asarray(test_loader.dataset.label, dtype=int)


    roc_test = roc_auc_score(np.asarray(test_loader.dataset.label,dtype=int),loss_test)

    import numpy as np
    from sklearn.metrics import precision_recall_curve, f1_score



    y_scores = loss_test
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        f1_scores[np.isnan(f1_scores)] = 0
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]
    best_precision = precisions[best_index]
    best_recall = recalls[best_index]
    best_f1_score = f1_scores[best_index]

    print(f"F1 Score: {best_f1_score}")
    print(f"Precision: {best_precision}")
    print(f"Recall: {best_recall}")
    pr_auc = auc(recalls, precisions)
    print("The AUCROC score on {} dataset is {}".format(args.name, roc_test))
    print("The AUCPR score on {} dataset is {}".format(args.name, pr_auc))

    y_pred=y_scores>=best_threshold
    P_ew, R_ew, F1_ew = compute_event_wise_metrics(
        y_true, y_pred, gt_intervals=None)#
    print(f"P_ew: {P_ew}, R_ew: {R_ew}, F1_ew: {F1_ew}")




