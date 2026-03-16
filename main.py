import argparse
from models.ALDM import ALDM
from torch.optim import lr_scheduler
from utils import *


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, 
                    default='./Dataset/Data/input/SWaT_Dataset_Attack_v0.csv', help='Location of datasets.')
parser.add_argument('--output_dir', type=str, 
                    default='./checkpoint/')
parser.add_argument('--name',default='SWaT', help='the name of dataset',choices=["SWaT", "MSL", "PSM", "TEPE"])

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
parser.add_argument('--outdim_scl', type=int, default=320, help='Dimension of output layer for latent encoding.')
parser.add_argument('--hiddendim_scl', type=int, default=128, help='Dimension of hidden layers for latent encoding.')
parser.add_argument('--soft_instance', type=bool, default=False)
parser.add_argument('--soft_temporal', type=bool, default=True)

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--window_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=0.002, help='Learning rate.')
parser.add_argument('--max_lr', type=float, default=0.004, help='Max Learning rate.')
parser.add_argument('--loss_weight', type=float, default=0.5)



args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


for seed in range(15,20):
    args.seed = seed
    print(args)
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    #%%
    print("Loading dataset")
    print(args.name)
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
    elif args.name == 'TEPE':
        train_loader, val_loader, test_loader, n_sensor = loader_tepe(args.name, \
                                                                    args.batch_size, args.window_size, args.stride_size, args.train_split)




    #%%
    model = ALDM(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.window_size, n_sensor,
                  args.outdim_scl, args.hiddendim_scl, args.soft_instance, args.soft_temporal, args.dim_ml, dropout=0.0,
                  model=args.model, batch_norm=args.batch_norm)
    model = model.to(device)

    #%%
    from torch.nn.utils import clip_grad_value_
    save_path = os.path.join(args.output_dir,args.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    loss_best = 100
    roc_max = 0

    lr = args.lr
    a=args.loss_weight
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'weight_decay': args.weight_decay},
    ], lr=lr, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                        steps_per_epoch=len(train_loader),
                                        pct_start=0.2,
                                        epochs=args.epochs,
                                        max_lr=args.max_lr)#scheduler

    for epoch in range(args.epochs):
        print(epoch)
        loss_train = []

        model.train()
        for x,_,_ in train_loader:
            x = x.to(device)

            optimizer.zero_grad()
            loss, loss_scl,_= model.test(x, )
            loss_nz = loss.mean()
            total_loss = -loss_nz*(1-a) + 0.01 * loss_scl*a
            total_loss.backward()
            clip_grad_value_(model.parameters(), 1)
            optimizer.step()
            loss_train.append(total_loss.item())




        loss_test = []
        with torch.no_grad():
            for x,_,_ in test_loader:

                x = x.to(device)
                loss,_ ,_= model.test(x, )
                loss=loss.cpu().numpy()
                loss_test.append(-loss)

        loss_test = np.concatenate(loss_test)

        y_true = np.asarray(test_loader.dataset.label, dtype=int)
        roc_test = roc_auc_score(y_true, loss_test)
        if roc_max < roc_test:
            roc_max = roc_test
            torch.save({
            'model': model.state_dict(),
            }, f"{save_path}/model_seed_{args.seed}.pth")

        roc_max = max(roc_test, roc_max)
        print(roc_max)

