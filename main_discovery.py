import torch
import numpy as np
from torch.utils.data import DataLoader
from model import *
from discovery import *
from utils import *
from dataset import get_dataset
from parser_utils import get_discovery_args

save_model_path = './saved_models'

if __name__ == '__main__':
    args = get_discovery_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args = vars(args)

    train_dataset, _, args = get_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = MLP(**args).to(args['device'])
    model.load_state_dict(torch.load(f"{save_model_path}/{args['save_dir']}/model_{args['epoch']}.pt", weights_only=True))

    S, Vh = symmetry_discovery(model=model, train_loader=train_loader, **args)
    Vh = basis_sparsification(Q=Vh.T, **args).T
    vis(S=S, Vh=Vh, task=args['task'], sample=args['sample'])
    print(S)