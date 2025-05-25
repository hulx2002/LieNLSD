# modified from https://github.com/Rose-STL-Lab/LieGAN/blob/master/main_lagan.py

import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from dataset import *
from gan import LieGenerator, LieDiscriminator, LieDiscriminatorEmb
from train import train_lie_gan, train_lie_gan_incremental


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model & training settings
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--lr_g', type=float, default=1e-3)
    parser.add_argument('--reg_type', type=str, default='cosine')
    parser.add_argument('--lamda', type=float, default=1e-2)
    parser.add_argument('--p_norm', type=float, default=2)
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--model', type=str, default='lie')
    parser.add_argument('--coef_dist', type=str, default='normal')
    parser.add_argument('--g_init', type=str, default='random')
    parser.add_argument('--sigma_init', type=float, default=1)
    parser.add_argument('--uniform_max', type=float, default=1)
    parser.add_argument('--normalize_Li', action='store_true')
    parser.add_argument('--n_channel', type=int, default=1)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--incremental', action='store_true')
    # dataset settings
    parser.add_argument('--task', type=str, default='traj_pred')
    parser.add_argument('--n_component', type=int, default=1)
    parser.add_argument('--noise', type=float, default=0.0)
    # run settings
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='saved_model')
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    if args.task == 'top':
        dataset = TopTagging(n_component=args.n_component, noise=args.noise)
        n_dim = 4
        n_channel = args.n_channel
        n_component = args.n_component
        d_input_size = n_dim * n_component
        n_class = 2
        emb_size = 32
    elif args.task in ['heat', 'burger', 'kdv', 'wave', 'rd', 'schrodinger']:
        dataset = PDEDataset(pde_name=args.task, mode='train')
        n_dim = len(dataset[0][0]) + len(dataset[0][1])
        n_channel = args.n_channel
        d_input_size = n_dim
    else:
        raise NotImplementedError

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize generator and discriminator
    generator = LieGenerator(n_dim, n_channel, args).to(args.device)
    if args.task == 'top':
        discriminator = LieDiscriminatorEmb(d_input_size, n_class, emb_size).to(args.device)
    else:
        discriminator = LieDiscriminator(d_input_size).to(args.device)
    if args.model == 'lie':  # fix the coefficient distribution
        generator.mu.requires_grad = False
        generator.sigma.requires_grad = False
    elif args.model == 'lie_subgrp':  # fix the generator
        generator.Li.requires_grad = False

    # Train
    train_fn = train_lie_gan if not args.incremental else train_lie_gan_incremental
    train_fn(
        generator,
        discriminator,
        dataloader,
        args.num_epochs,
        args.lr_d,
        args.lr_g,
        args.reg_type,
        args.lamda,
        args.p_norm,
        args.mu,
        args.eta,
        args.device,
        task=args.task,
        save_path=f'{args.save_path}/{args.task}_{args.seed}/',
        print_every=args.print_every,
    )
