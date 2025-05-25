import torch
import numpy as np
from torch.utils.data import DataLoader
from model import *
from train import *
from dataset import get_dataset
from parser_utils import get_train_args

if __name__ == '__main__':
    args = get_train_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args = vars(args)

    train_dataset, val_dataset, args = get_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)

    model = MLP(**args).to(args['device'])

    train_model(model=model, train_loader=train_loader, test_loader=val_loader, **args)