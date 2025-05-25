import math
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def basis_sparsification(Q, epsilon1, epsilon2, device, log_interval, **kwargs):
    
    def soft_threshold(x, epsilon):
        return torch.sign(x) * torch.clamp(torch.abs(x) - epsilon, min=0.0)

    Q = Q.to(device)
    n, d = Q.shape
    beta = min(n, d)
    beta_max = 1e10
    rho0 = 1.9
    etaA = 1.02 * torch.linalg.norm(Q, ord=2) ** 2
    etaB = 1.02
    R = torch.eye(d, device=device)
    Z = Q @ R
    Lambda = torch.zeros_like(Q)
    k = 0
    while True:
        R_pre = R.clone()
        Z_pre = Z.clone()
        tempR = R - (Q.T @ (Lambda + beta * (Q @ R - Z))) / (beta * etaA)
        tempR = Q.T @ Z / etaA - Q.T @ Lambda / (beta * etaA)
        U, S, Vh = torch.linalg.svd(tempR.to(torch.float64))
        U, Vh = U.to(torch.float32), Vh.to(torch.float32)
        R = U @ Vh
        tempZ = Z + (Lambda + beta * (Q @ R - Z)) / (beta * etaB)
        Z = soft_threshold(tempZ, 1.0 / (beta * etaB))
        Lambda = Lambda + beta * (Q @ R - Z)
        condition = beta * max(math.sqrt(etaA) * torch.linalg.norm(R - R_pre, ord=float('inf')).item(), math.sqrt(etaB) * torch.linalg.norm(Z - Z_pre, ord=float('inf')).item())
        if condition < epsilon2:
            rho = rho0
        else:
            rho = 1.0
        beta = min(beta_max, rho * beta)
        loss = torch.sum(torch.abs(Q @ R)).item()
        if (k + 1) % log_interval == 0:
            print(f'Epoch {k}, loss: {loss:.4f}')
        k += 1
        if torch.linalg.norm(Q @ R - Z, ord=float('inf')) < epsilon1 and condition <= epsilon2:
            return Q @ R

save_vis_path = './vis'

def vis(S, Vh, task, sample):
    S = S.cpu().numpy()
    plt.plot(range(S.shape[0]-Vh.shape[0], S.shape[0]+1), S[S.shape[0]-Vh.shape[0]-1:S.shape[0]])
    plt.xticks(range(S.shape[0]-Vh.shape[0], S.shape[0]+1))
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Singular Value')
    if not os.path.exists(f'{save_vis_path}/{task}_{sample}'):
        os.makedirs(f'{save_vis_path}/{task}_{sample}')
    plt.savefig(f'{save_vis_path}/{task}_{sample}/singular_value.png', bbox_inches='tight')
    plt.clf()

    n = Vh.shape[0]
    if task == 'top':
        W = Vh.reshape(n, 4, 4)
    elif task in ['heat', 'burger', 'kdv']:
        W = Vh.reshape(n, 3, 10)
    elif task == 'wave':
        W = Vh.reshape(n, 4, 15)
    elif task in ['rd', 'schrodinger']:
        W = Vh.reshape(n, 5, 21)
    torch.save(W, f'{save_vis_path}/{task}_{sample}/generator.pt')
    W = W.cpu().numpy()
    for i in range(n):
        plt.imshow(W[i])
        if task == 'top':
            plt.colorbar()
            plt.xticks(ticks=np.arange(4), labels=[r'$p^0$', r'$p^1$', r'$p^2$', r'$p^3$'])
            plt.yticks(ticks=np.arange(4), labels=[r'$\partial_{p^0}$', r'$\partial_{p^1}$', r'$\partial_{p^2}$', r'$\partial_{p^3}$'])
        elif task in ['heat', 'burger', 'kdv']:
            plt.colorbar(shrink=0.5)
            plt.xticks(ticks=np.arange(10), labels=[r'$1$', r'$t$', r'$x$', r'$u$', r'$t^2$', r'$x^2$', r'$u^2$', r'$tx$', r'$tu$', r'$xu$'])
            plt.yticks(ticks=np.arange(3), labels=[r'$\partial_t$', r'$\partial_x$', r'$\partial_u$'])
        elif task == 'wave':
            plt.colorbar(shrink=0.5)
            plt.xticks(ticks=np.arange(15), labels=[r'$1$', r'$t$', r'$x$', r'$y$', r'$u$', r'$t^2$', r'$x^2$', r'$y^2$', r'$u^2$', r'$tx$', r'$ty$', r'$tu$', r'$xy$', r'$xu$', r'$yu$'])
            plt.yticks(ticks=np.arange(4), labels=[r'$\partial_t$', r'$\partial_x$', r'$\partial_y$', r'$\partial_u$'])
        elif task in ['rd', 'schrodinger']:
            plt.colorbar(shrink=0.5)
            plt.xticks(ticks=np.arange(21), labels=[r'$1$', r'$t$', r'$x$', r'$y$', r'$u$', r'$v$', r'$t^2$', r'$x^2$', r'$y^2$', r'$u^2$', r'$v^2$', r'$tx$', r'$ty$', r'$tu$', r'$tv$', r'$xy$', r'$xu$', r'$xv$', r'$yu$', r'$yv$', r'$uv$'])
            plt.yticks(ticks=np.arange(5), labels=[r'$\partial_t$', r'$\partial_x$', r'$\partial_y$', r'$\partial_u$', r'$\partial_v$'])
        plt.tick_params(axis='both', which='both', length=0)
        plt.savefig(f'{save_vis_path}/{task}_{sample}/generator_{S.shape[0] - n + i + 1}.png', bbox_inches='tight')
        plt.clf()
