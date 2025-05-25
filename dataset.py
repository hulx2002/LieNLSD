import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

# modified from https://github.com/Rose-STL-Lab/symmetry-ode-discovery/blob/main/dataset.py
data_path = './data'

def get_dataset(args):
    if args['task'] == 'top':
        train_dataset = TopTagging(path='./data/top/train.h5', n_component=20)
        val_dataset = TopTagging(path='./data/top/val.h5', n_component=20)
        args['input_dim'] = 80
        args['output_dim'] = 1
        args['classify'] = True
    elif args['task'] == 'burger':
        train_dataset = PDEDataset(pde_name='burger', mode='train')
        val_dataset = PDEDataset(pde_name='burger', mode='val')
        args['input_dim'] = 3
        args['output_dim'] = 1
        args['classify'] = False
    elif args['task'] == 'wave':
        train_dataset = PDEDataset(pde_name='wave', mode='train')
        val_dataset = PDEDataset(pde_name='wave', mode='val')
        args['input_dim'] = 6
        args['output_dim'] = 1
        args['classify'] = False
    elif args['task'] in ['schrodinger']:
        train_dataset = PDEDataset(pde_name='schrodinger', mode='train')
        val_dataset = PDEDataset(pde_name='schrodinger', mode='val')
        args['input_dim'] = 12
        args['output_dim'] = 2
        args['classify'] = False
    elif args['task'] == 'heat':
        train_dataset = PDEDataset(pde_name='heat', mode='train')
        val_dataset = PDEDataset(pde_name='heat', mode='val')
        args['input_dim'] = 3
        args['output_dim'] = 1
        args['classify'] = False
    elif args['task'] == 'kdv':
        train_dataset = PDEDataset(pde_name='kdv', mode='train')
        val_dataset = PDEDataset(pde_name='kdv', mode='val')
        args['input_dim'] = 4
        args['output_dim'] = 1
        args['classify'] = False
    elif args['task'] in ['rd']:
        train_dataset = PDEDataset(pde_name='rd', mode='train')
        val_dataset = PDEDataset(pde_name='rd', mode='val')
        args['input_dim'] = 12
        args['output_dim'] = 2
        args['classify'] = False
    else:
        raise NotImplementedError
    return train_dataset, val_dataset, args

class PDEDataset(Dataset):
    def __init__(self, path=f'{data_path}', pde_name='heat', mode='train'):
        super().__init__()
        try:
            print(f'Loading existing {pde_name} {mode} data...')
            data = {}
            if pde_name in ['heat', 'burger']:
                keys = ['t', 'x', 'u', 'dudx', 'dudxdx', 'dudt', 'dudtdx']
            elif pde_name == 'kdv':
                keys = ['t', 'x', 'u', 'dudx', 'dudxdx', 'dudxdxdx', 'dudt', 'dudtdx', 'dudtdxdx']
            elif pde_name == 'wave':
                keys = ['t', 'x', 'y', 'u', 'dudt', 'dudx', 'dudy', 'dudtdt', 'dudxdx', 'dudydy', 'dudtdx', 'dudtdy', 'dudxdy']
            elif pde_name in ['rd', 'schrodinger']:
                keys = ['t', 'x', 'y', 'u', 'dudt', 'dudx', 'dudy', 'dudxdx', 'dudydy', 'dudtdx', 'dudtdy', 'dudxdy', 'v', 'dvdt', 'dvdx', 'dvdy', 'dvdxdx', 'dvdydy', 'dvdtdx', 'dvdtdy', 'dvdxdy']
            for key in keys:
                data[key] = torch.load(f'{path}/{pde_name}/{mode}-{key}.pt', weights_only=True)
        except FileNotFoundError:
            print(f'Load data failed.')
            exit()

        for key in data:
            data[key] = data[key].to(torch.float32)

        self.data = {}
        if pde_name in ['heat', 'burger', 'kdv']:
            self.data['t'] = data['t'].unsqueeze(0).unsqueeze(-1).expand(data['u'].shape).reshape(-1)
            self.data['x'] = data['x'].unsqueeze(0).unsqueeze(0).expand(data['u'].shape).reshape(-1)
        elif pde_name in ['wave', 'rd', 'schrodinger']:
            self.data['t'] = data['t'].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(data['u'].shape).reshape(-1)
            self.data['x'] = data['x'].unsqueeze(0).unsqueeze(0).expand(data['u'].shape).reshape(-1)
            self.data['y'] = data['y'].unsqueeze(0).unsqueeze(0).expand(data['u'].shape).reshape(-1)
        for key in data:
            if key not in ['t', 'x', 'y', 'z']:
                self.data[key] = data[key].reshape(-1)

    def __len__(self):
        return len(self.data['u'])
    
    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.data}

# modified from https://github.com/Rose-STL-Lab/LieGAN/blob/master/dataset.py
class TopTagging(torch.utils.data.Dataset):
    def __init__(self, path='./data/top/train.h5', flatten=False, n_component=3, noise=0.0):
        super().__init__()
        df = pd.read_hdf(path, key='table')
        df = df.to_numpy()
        self.X = df[:, :4*n_component]
        self.X = self.X * np.random.uniform(1-noise, 1+noise, size=self.X.shape)
        self.y = df[:, -1]
        self.X = torch.FloatTensor(self.X)
        if not flatten:
            self.X = self.X.reshape(-1, n_component, 4)
        self.y = torch.FloatTensor(self.y)
        self.len = self.X.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx]}