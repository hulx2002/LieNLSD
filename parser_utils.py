import argparse
import torch

def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='heat')

    parser.add_argument('--opt', type=str, default='Adan')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--activation', type=str, default='ReLU')

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='test')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    return args

def get_discovery_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='heat')

    parser.add_argument('--epsilon1', type=float, default=1e-4)
    parser.add_argument('--epsilon2', type=float, default=1e-5)

    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--activation', type=str, default='ReLU')
    parser.add_argument('--save_dir', type=str, default='test')
    parser.add_argument('--epoch', type=int, default=19)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample', type=int, default=0)

    args = parser.parse_args()

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    return args