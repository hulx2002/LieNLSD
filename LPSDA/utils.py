# modified from https://github.com/kogyeonghoon/learning-symmetry-from-scratch/blob/main/pde/utils.py

import os
from datetime import datetime
import torch
import numpy as np
import pandas as pd

import yaml

from torch.utils.data import Dataset

def default_experiment_name(args):
    if not hasattr(args, 'delta_exp'):
        return str(args.seed)
    elif args.n_transform == 1:
        return 'none' + str(args.seed)
    elif args.delta_exp == 'lps_lienlsd':
        return 'lienlsd' + str(args.seed)
    elif args.delta_exp == 'lps_learning':
        return 'learning' + str(args.seed)
    elif args.delta_exp == 'lps_gt':
        return 'gt' + str(args.seed)

data_path = '../data'

class PDEDataset(Dataset):
    def __init__(self, path=f'{data_path}', pde_name='heat', mode='train'):
        super().__init__()
        try:
            print(f'Loading existing {pde_name} {mode} data...')
            data = {}
            if pde_name in ['heat', 'burger']:
                keys = ['t', 'x', 'u', 'dudt', 'dudx', 'dudxdx']
            elif pde_name == 'kdv':
                keys = ['t', 'x', 'u', 'dudt', 'dudx', 'dudxdx', 'dudxdxdx']
            for key in keys:
                data[key] = torch.load(f'{path}/{pde_name}/{mode}-{key}.pt', weights_only=True)
        except FileNotFoundError:
            print(f'Load data failed.')
            exit()

        for key in data:
            data[key] = data[key].to(torch.float32)

        data['t'] = data['t'][::10]
        data['x'] = data['x'][:10]
        data['u'] = data['u'][:, ::10, :10]
        data['dudt'] = data['dudt'][:, ::10, :10]
        data['dudx'] = data['dudx'][:, ::10, :10]
        data['dudxdx'] = data['dudxdx'][:, ::10, :10]
        if pde_name == 'kdv':
            data['dudxdxdx'] = data['dudxdxdx'][:, ::10, :10]

        self.data = {}
        self.data['t'] = data['t'].unsqueeze(0).unsqueeze(-1).expand(data['u'].shape)
        self.data['x'] = data['x'].unsqueeze(0).unsqueeze(0).expand(data['u'].shape)
        self.data['u'] = data['u']
        self.data['dudt'] = data['dudt']
        self.data['dudx'] = data['dudx']
        self.data['dudxdx'] = data['dudxdx']
        if pde_name == 'kdv':
            self.data['dudxdxdx'] = data['dudxdxdx']

    def __len__(self):
        return self.data['u'].shape[0]
    
    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.data}

class AugmentedDataset(Dataset):
    def __init__(self,
                 data,
                 t_eff,
                 reject,
                 n_data,
                 n_transform,
                 p_original = 0):
        self.data = data
        self.t_eff = t_eff
        self.reject = reject
        self.n_data = n_data
        self.n_transform = n_transform
        self.p_original = p_original
        
    def __len__(self):
        return self.n_data
    
    def __getitem__(self, idx):
        if np.random.rand() < self.p_original:
            idx2 = 0
        else:
            idx2 = np.random.randint(self.n_transform)
        reject = self.reject[idx2, idx]
        if reject:
            idx2 = 0
        u = self.data[idx2, idx]
        t_eff = self.t_eff[idx2, idx]
        return u, t_eff

class GetPatch():
    def __init__(self, nx, nt, patch_size):
        self.nx = nx
        self.nt = nt
        self.patch_size = patch_size
        
    def __call__(self, xtu, patch_x, patch_t):
        xtu_patch = []
        for i, (x_, t_) in enumerate(zip(patch_x, patch_t)):
            xtu_patch.append(self.instancewise_patch(xtu[i], x_, t_))
        return torch.stack(xtu_patch, dim=0)
        
    def instancewise_patch(self, xtu_, x_, t_):
        if x_ <= self.nx - self.patch_size:
            return xtu_[t_:t_+self.patch_size, x_:x_+self.patch_size]
        else:
            return torch.cat((xtu_[t_:t_+self.patch_size, x_:],
                              xtu_[t_:t_+self.patch_size, :x_+self.patch_size-self.nx]),
                              dim=1)

class RandomSign():
    def __init__(self, n_delta, device):
        self.n_delta = n_delta
        self.device = device
    
    def set_sign(self):
        self.sign = (-1) ** torch.randint(0, 2, size=(1, 1, self.n_delta,), device=self.device)
    
    def apply(self, delta):
        return delta * self.sign

class LipschitzLoss():
    def __init__(self, tau):
        self.tau = tau
    
    def __call__(self, delta, xtu):
        dt_delta = torch.diff(delta, dim=1)
        dt_delta = torch.norm(dt_delta, dim=3)
        dt_xtu =  torch.diff(xtu, dim=1)
        dt_xtu = torch.norm(dt_xtu, dim=3)
        
        dt_delta = dt_delta / dt_xtu
        dt_delta = torch.clamp(dt_delta-self.tau, min=0)

        dx_delta = delta - torch.roll(delta, shifts=1, dims=2)
        dx_delta = torch.norm(dx_delta, dim=3)
        dx_xtu =  xtu - torch.roll(xtu, shifts=1, dims=2)
        # for periodic boundary
        idx = dx_xtu > 0.5
        idx[:, :, :, 1:, :] = False
        dx_xtu[idx] = dx_xtu[idx] - 1
        idx = dx_xtu < -0.5
        idx[:, :, :, 1:, :] = False
        dx_xtu[idx] = dx_xtu[idx] + 1
        
        dx_xtu = torch.norm(dx_xtu, dim=3)
        
        dx_delta = dx_delta / dx_xtu
        dx_delta = torch.clamp(dx_delta - self.tau, min=0)
        
        return dt_delta.mean(dim=(0,1,2)) + dx_delta.mean(dim=(0,1,2))

class SobolevLoss():
    def __init__(self, k, nx, dx):
        self.k = k
        self.nx = nx
        self.dx = dx
    
    def __call__(self, delta):
        freq = torch.fft.fftfreq(self.nx, self.dx).to(delta.device)
        delta_hat = torch.fft.fft(delta, dim=2)
        sobolev = ((delta_hat.abs() ** 2) * ((1 + freq ** 2) ** self.k - 1)[None, None, :, None, None]).mean(dim=2)
        return sobolev.log().mean(dim=(0,1,2))

def init_path(experiment_name, exp_name, args, subdirs=[]):
    os.makedirs(experiment_name, exist_ok=True)
    
    exp_path = os.path.join(experiment_name, exp_name)
    os.makedirs(exp_path, exist_ok=True)

    outfile = os.path.join(exp_path, 'log.txt')
    with open(outfile, 'w') as f:
        f.write(f'Experiment {exp_name}' + '\n')
    with open(os.path.join(exp_path, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    for subdir in subdirs:
        subdir_path = os.path.join(exp_path, subdir)
        os.makedirs(subdir_path, exist_ok=True)

    return exp_path, outfile

class Writer():
    def __init__(self, outfile):
        self.outfile = outfile
    def __call__(self, outstr):
        with open(self.outfile, 'a') as f:
            f.write(outstr + '\n')
        print(outstr)

class MetricTracker():
    # gathers metrics, take averages and record them, export to pandas
    # supports float & torch tensor (1-dim)
    
    def __init__(self,metric_names):
        self.metric_names = metric_names
        self.metric_history = {n:[] for n in self.metric_names}
        self.initialize()
        self.step = 0
        
    def initialize(self):
        self.metric_collect = {n:[] for n in self.metric_names}
        
    def update(self,new_metric):
        for n in self.metric_names:
            if n in new_metric.keys():
                self.metric_collect[n].append(new_metric[n])
            
    def aggregate(self):
        new_dict = dict()
        for n in self.metric_names:
            if len(self.metric_collect[n]) == 0:
                continue
            if isinstance(self.metric_collect[n][0], float):
                metric_mean = np.mean(self.metric_collect[n])
            elif type(self.metric_collect[n][0]) == torch.Tensor:
                metric_mean = torch.stack(self.metric_collect[n], dim=1).mean(dim=1).numpy()
            new_dict[n] = metric_mean
            self.metric_history[n].append(metric_mean)
        self.initialize()
        self.step += 1
        return new_dict
    
    def to_pandas(self):
        if self.step == 0:
            return pd.DataFrame()
        else:
            df_list = []
            for n in self.metric_names:
                if len(self.metric_history[n]) == 0:
                    continue
                if not isinstance(self.metric_history[n][0], np.ndarray):
                    df = pd.Series(self.metric_history[n], name=n)
                    df_list.append(df)
                else:
                    n_cols = len(self.metric_history[n][0])
                    df = pd.DataFrame(self.metric_history[n], columns=[n + str(i) for i in range(n_cols)])
                    df_list.append(df)
            return pd.concat(df_list, axis=1)
