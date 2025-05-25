# modified from https://github.com/kogyeonghoon/learning-symmetry-from-scratch/blob/main/pde/fno_train_helper.py

import torch
from typing import Tuple
from torch import nn
from torch.utils.data import DataLoader
import argparse

def bootstrap(x: torch.Tensor, Nboot: int=64, binsize: int=1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bootstrapping mean or median to obtain standard deviation.
    Args:
        x (torch.Tensor): input tensor, which contains all the results on the different trajectories of the set at hand
        Nboot (int): number of bootstrapping steps, 64 is quite common default value
        binsize (int):
    Returns:
        torch.Tensor: bootstrapped mean/median of input
        torch.Tensor: bootstrapped variance of input
    """
    boots = []
    x = x.reshape(-1, binsize, *x.shape[1:])
    for i in range(Nboot):
        boots.append(torch.mean(x[torch.randint(len(x), (len(x),))], axis=(0, 1)))
    return torch.tensor(boots).mean(), torch.tensor(boots).std()

class DataCreator(nn.Module):
    """
    Helper class to construct input data and labels.
    """
    def __init__(self,
                 time_history,
                 time_future
                 ):
        super().__init__()
        self.time_history = time_history
        self.time_future = time_future

    def create_data(self, datapoints: torch.Tensor, start_time: list) -> Tuple[torch.Tensor, torch.Tensor]:
        data = []
        labels = []
        # Loop over batch and different starting points
        # For every starting point, we take the number of time_history points as training data
        # and the number of time future data as labels
        for (dp, start) in zip(datapoints, start_time):
            end_time = start + self.time_history
            d = dp[start:end_time]
            target_start_time = end_time
            target_end_time = target_start_time + self.time_future
            l = dp[target_start_time:target_end_time]

            data.append(d.unsqueeze(dim=0))
            labels.append(l.unsqueeze(dim=0))

        return torch.cat(data, dim=0), torch.cat(labels, dim=0)
    

def test_timestep_losses(model: torch.nn.Module,
                         batch_size: int,
                         loader: DataLoader,
                         data_creator: DataCreator,
                         criterion: torch.nn.modules.loss,
                         device: torch.cuda.device = "cpu",
                         print_out = print) -> None:
    # Length of trajectory
    time_resolution = next(iter(loader))['u'].shape[1]
    # Max number of previous points solver can eat
    reduced_time_resolution = time_resolution - data_creator.time_history
    # Number of future points to predict
    max_start_time = reduced_time_resolution - data_creator.time_future
    # The first time steps are used for data augmentation according to time translate
    # We ignore these timesteps in the testing
    start_time = [t for t in range(data_creator.time_history, max_start_time + 1, data_creator.time_future)]
    for start in start_time:
        losses = []
        for data in loader:
            u = data['u']
            with torch.no_grad():
                end_time = start + data_creator.time_history
                target_end_time = end_time + data_creator.time_future
                data = u[:, start:end_time]
                labels = u[:, end_time:target_end_time]
                data, labels = data.to(device), labels.to(device)

                data = data.permute(0, 2, 1)
                pred = model(data)
                loss = criterion(pred.permute(0, 2, 1), labels)
                loss = loss.sum()
                losses.append(loss / batch_size)

        losses = torch.stack(losses)
        print_out(f'Input {start} - {start + data_creator.time_history}, mean loss {torch.mean(losses)}')

def test_unrolled_losses(model: torch.nn.Module,
                         batch_size: int,
                         loader: DataLoader,
                         data_creator: DataCreator,
                         criterion: torch.nn.modules.loss,
                         device: torch.cuda.device = "cpu") -> (torch.Tensor, torch.Tensor): # type: ignore
    time_resolution = next(iter(loader))['u'].shape[1]
    # Max number of previous points solver can eat
    reduced_time_resolution = time_resolution - data_creator.time_history
    # Number of future points to predict
    max_start_time = reduced_time_resolution - data_creator.time_future

    losses, nlosses = [], []
    for data in loader:
        u = data['u']
        nx = u.shape[2]
        losses_tmp, nlosses_tmp = [], []
        with torch.no_grad():
            # the first time steps are used for data augmentation according to time translate
            # we ignore these timesteps in the testing
            for start in range(data_creator.time_history, max_start_time+1, data_creator.time_future):
                end_time = start + data_creator.time_history
                target_end_time = end_time + data_creator.time_future
                if start == data_creator.time_history:
                    data = u[:, start:end_time].to(device)
                    data = data.permute(0, 2, 1)
                else:
                    data = torch.cat([data, pred], -1)
                    data = data[..., -data_creator.time_history:]
                labels = u[:, end_time:target_end_time].to(device)

                pred = model(data)

                loss = criterion(pred.permute(0, 2, 1), labels)
                nlabels = torch.mean(labels ** 2, dim=-1, keepdim=True)
                nloss = loss / nlabels
                loss, nloss = loss.sum(), nloss.sum()
                loss, nloss = loss / nx / batch_size, nloss / nx / batch_size
                losses_tmp.append(loss)
                nlosses_tmp.append(nloss)

        losses.append(torch.sum(torch.stack(losses_tmp)))
        nlosses.append(torch.sum(torch.stack(nlosses_tmp)))

    losses = torch.stack(losses)
    nlosses = torch.stack(nlosses)

    return losses, nlosses


def test(args: argparse,
         model: torch.nn.Module,
         loader: DataLoader,
         data_creator: DataCreator,
         criterion: torch.nn.modules.loss,
         device: torch.cuda.device="cpu",
         print_out=print) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()

    # Check the losses for different timesteps (one forward prediction step)
    losses = test_timestep_losses(model=model,
                                  batch_size=args.test_batch_size,
                                  loader=loader,
                                  data_creator=data_creator,
                                  criterion=criterion,
                                  device=device,
                                  print_out = print_out)

    # Test the unrolled losses (full trajectory)
    losses, nlosses = test_unrolled_losses(model=model,
                                           batch_size=args.test_batch_size,
                                           loader=loader,
                                           data_creator=data_creator,
                                           criterion=criterion,
                                           device=device)


    mean, std = bootstrap(losses, 64, 1)
    nmean, nstd = bootstrap(nlosses, 64, 1)
    print_out(f'Unrolled forward losses: {mean:.4f} +- {std:.4f}')
    print_out(f'Unrolled forward losses (normalized): {nmean:.8f} +- {nstd:.8f}')
    return mean, std, nmean, nstd
