import torch
import numpy as np
from utils import *
import pickle
import pandas as pd

data_path = '../data'

# modified from https://github.com/Rose-STL-Lab/LieGAN/blob/master/dataset.py
class TopTagging(torch.utils.data.Dataset):
    def __init__(self, path='../data/top/train.h5', flatten=False, n_component=3, noise=0.0):
        super().__init__()
        df = pd.read_hdf(path, key='table')
        df = df.to_numpy()
        self.X = df[:, :4*n_component]
        self.X = self.X * np.random.uniform(1-noise, 1+noise, size=self.X.shape)
        self.y = df[:, -1]
        self.X = torch.FloatTensor(self.X)
        if not flatten:
            self.X = self.X.reshape(-1, n_component, 4)
        self.y = torch.LongTensor(self.y)
        self.len = self.X.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# modified from https://github.com/Rose-STL-Lab/symmetry-ode-discovery/blob/main/dataset.py
class PDEDataset(torch.utils.data.Dataset):
    def __init__(self, path=f'{data_path}', pde_name='heat', mode='train'):
        super().__init__()
        try:
            print(f'Loading existing {pde_name} {mode} data...')
            data = {}
            if pde_name in ['heat', 'burger', 'kdv']:
                keys = ['t', 'x', 'u']
            elif pde_name == 'wave':
                keys = ['t', 'x', 'y', 'u']
            elif pde_name in ['rd', 'schrodinger']:
                keys = ['t', 'x', 'y', 'u', 'v']
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
        elif pde_name == 'wave':
            self.data['t'] = data['t'].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(data['u'].shape).reshape(-1)
            self.data['x'] = data['x'].unsqueeze(0).unsqueeze(0).expand(data['u'].shape).reshape(-1)
            self.data['y'] = data['y'].unsqueeze(0).unsqueeze(0).expand(data['u'].shape).reshape(-1)
        elif pde_name in ['rd', 'schrodinger']:
            self.data['t'] = data['t'].unsqueeze(-1).unsqueeze(-1).expand(data['u'].shape).reshape(-1)
            self.data['x'] = data['x'].unsqueeze(0).expand(data['u'].shape).reshape(-1)
            self.data['y'] = data['y'].unsqueeze(0).expand(data['u'].shape).reshape(-1)
        for key in data:
            if key not in ['t', 'x', 'y', 'z']:
                self.data[key] = data[key].reshape(-1)
        if pde_name in ['heat', 'burger', 'kdv']:
            self.X = torch.stack([self.data['t'], self.data['x']], dim=1)
            self.y = self.data['u'].unsqueeze(1)
        elif pde_name == 'wave':
            self.X = torch.stack([self.data['t'], self.data['x'], self.data['y']], dim=1)
            self.y = self.data['u'].unsqueeze(1)
        elif pde_name in ['rd', 'schrodinger']:
            self.X = torch.stack([self.data['t'], self.data['x'], self.data['y']], dim=1)
            self.y = torch.stack([self.data['u'], self.data['v']], dim=1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
