import torch
import numpy as np
import argparse

def subspace_distance(Q1, Q2):
    Q1, _ = torch.linalg.qr(Q1)
    Q2, _ = torch.linalg.qr(Q2)
    _, S, _ = torch.linalg.svd((Q1.T @ Q2).to(torch.float64))
    theta = torch.arccos(torch.clamp(S.to(torch.float32), -1.0, 1.0))
    distance = torch.sqrt(torch.sum(theta ** 2)).item()
    return distance

def ground_truth(task):
    if task == 'top':
        W = torch.zeros(7, 4, 4)
        W[0, 0, 0] = W[0, 1, 1] = W[0, 2, 2] = W[0, 3, 3] = 1
        W[1, 1, 2] = 1
        W[1, 2, 1] = -1
        W[2, 1, 3] = 1
        W[2, 3, 1] = -1
        W[3, 2, 3] = 1
        W[3, 3, 2] = -1
        W[4, 0, 1] = W[4, 1, 0] = 1
        W[5, 0, 2] = W[5, 2, 0] = 1
        W[6, 0, 3] = W[6, 3, 0] = 1
    elif task == 'burger':
        W = torch.zeros(2, 3, 3)
        W[0, 0, 0] = 2
        W[0, 1, 1] = 1
        W[1, 1, 0] = 2
        W[1, 2, 1] = -1
    elif task == 'wave':
        W = torch.zeros(8, 4, 4)
        W[0, 3, 0] = 1
        W[1, 3, 1] = 1
        W[2, 3, 2] = 1
        W[3, 3, 3] = 1
        W[4, 1, 2] = 1
        W[4, 2, 1] = -1
        W[5, 1, 0] = W[5, 0, 1] = 1
        W[6, 2, 0] = W[6, 0, 2] = 1
        W[7, 0, 0] = W[7, 1, 1] = W[7, 2, 2] = 1
    elif task == 'schrodinger':
        W = torch.zeros(3, 5, 5)
        W[0, 1, 2] = 1
        W[0, 2, 1] = -1
        W[1, 3, 4] = 1
        W[1, 4, 3] = -1
        W[2, 0, 0] = 2
        W[2, 1, 1] = W[2, 2, 2] = 1
        W[2, 3, 3] = W[2 ,4, 4] = -1
    elif task == 'heat':
        W = torch.zeros(3, 3, 3)
        W[0, 2, 1] = 1
        W[1, 2, 2] = 1
        W[2, 1, 1] = 1
        W[2, 0, 0] = 2
    elif task == 'kdv':
        W = torch.zeros(1, 3, 3)
        W[0, 1, 1] = 1
        W[0, 0, 0] = 3
        W[0, 2, 2] = -2
    elif task == 'rd':
        W = torch.zeros(2, 5, 5)
        W[0, 1, 2] = 1
        W[0, 2, 1] = -1
        W[1, 3, 4] = 1
        W[1, 4, 3] = -1
    Q2 = W.reshape(W.shape[0], -1).T
    return Q2

def evaluate(generator, task):
    Q1 = generator.reshape(generator.shape[0], -1).T
    Q2 = ground_truth(task)
    distance = subspace_distance(Q1, Q2)
    return distance

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='top')
    args = parser.parse_args()

    LieNLSD = []
    LieGAN = []
    if args.task == 'top':
        index1 = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6]
        ])
        index2 = torch.arange(0, 4)
    elif args.task == 'burger':
        index1 = torch.tensor([
            [1, 2],
            [1, 2],
            [1, 2]
        ])
        index2 = torch.arange(1, 4)
    elif args.task == 'wave':
        index1 = torch.tensor([
            [7, 8, 9, 11, 12, 13, 14, 15],
            [7, 9, 10, 11, 12, 13, 14, 15],
            [6, 7, 8, 11, 12, 13, 14, 16]
        ])
        index2 = torch.arange(1, 5)
    elif args.task == 'schrodinger':
        index1 = torch.tensor([
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2]
        ])
        index2 = torch.arange(1, 6)
    elif args.task == 'heat':
        index1 = torch.tensor([
            [1, 3, 4],
            [2, 3, 4],
            [2, 3, 4]
        ])
        index2 = torch.arange(1, 4)
    elif args.task == 'kdv':
        index1 = torch.tensor([
            [0], [0], [0]
        ])
        index2 = torch.arange(1, 4)
    elif args.task == 'rd':
        index1 = torch.tensor([
            [0, 1],
            [0, 1],
            [0, 1]
        ])
        index2 = torch.arange(1, 6)
    for i in range(3):
        generator = torch.load(f'vis/{args.task}_{i}/generator.pt', weights_only=True).cpu()
        generator = generator[index1[i]]
        generator = generator[:, :, index2]
        LieNLSD.append(evaluate(generator, args.task))
        Li = torch.load(f'LieGAN/saved_model/{args.task}_{i}/Li_99.pt', weights_only=True).cpu()
        LieGAN.append(evaluate(Li, args.task))
    print(f'LieNLSD: d0={LieNLSD[0]}, d1={LieNLSD[1]}, d2={LieNLSD[2]}, d_mean={np.mean(LieNLSD)}, d_std={np.std(LieNLSD)}')
    print(f'LieGAN: d0={LieGAN[0]}, d1={LieGAN[1]}, d2={LieGAN[2]}, d_mean={np.mean(LieGAN)}, d_std={np.std(LieGAN)}')
    

