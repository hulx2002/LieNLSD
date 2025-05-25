import numpy as np
import torch
from tqdm import trange
from functools import partial
import argparse

def generate_random_ics(n_ics=200, L=20, n=100, n_terms=10):
    dx = L / (n + 1)
    x2 = np.linspace(-L / 2, L / 2, n + 1)
    x = x2[:n]
    initial_conditions = []
    for _ in range(n_ics):
        a0 = np.random.randn()
        u0 = a0 / 2
        for n in range(1, n_terms + 1):
            an = np.random.randn()
            bn = np.random.randn()
            u0 += an * np.cos(2 * n * np.pi * x / L) + bn * np.sin(2 * n * np.pi * x / L)
        initial_conditions.append(u0)
    return np.array(initial_conditions), x, dx

def burger(dudx, dudxdx):
    dudt = dudxdx + dudx ** 2
    return dudt

def space_derivative(u, dx):
    dudx = (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / (2 * dx)
    dudxdx = (np.roll(u, -1, axis=-1) - 2 * u + np.roll(u, 1, axis=-1)) / (dx ** 2)
    return dudx, dudxdx

def gen_data(pde, init_fn, n_ics=200, L=20, n=100, n_terms=10, dt=0.002, num_steps=1000):
    t = np.arange(0, dt * num_steps, dt)
    u0, x, dx = init_fn(n_ics=n_ics, L=L, n=n, n_terms=n_terms)
    u = np.zeros((num_steps, *u0.shape))
    u[0] = u0
    for i in trange(num_steps):
        dudx1, dudxdx1 = space_derivative(u[i], dx)
        dudt1 = pde(dudx1, dudxdx1)
        if i == num_steps - 1:
            break
        k1 = dt * dudt1
        dudx2, dudxdx2 = space_derivative(u[i] + 0.5 * k1, dx)
        dudt2 = pde(dudx2, dudxdx2)
        k2 = dt * dudt2
        dudx3, dudxdx3 = space_derivative(u[i] + 0.5 * k2, dx)
        dudt3 = pde(dudx3, dudxdx3)
        k3 = dt * dudt3
        dudx4, dudxdx4 = space_derivative(u[i] + k3, dx)
        dudt4 = pde(dudx4, dudxdx4)
        k4 = dt * dudt4
        u[i + 1] = u[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    u = np.transpose(u, (1, 0, 2))
    return t, x, u

def compute_derivative(t, x, u):
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    dudx, dudxdx = space_derivative(u, dx)
    dudt = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    dudtdx, _ = space_derivative(dudt, dx)
    t = t[1:-1]
    x = x[1:-1]
    u = u[:, 1:-1, 1:-1]
    dudx = dudx[:, 1:-1, 1:-1]
    dudxdx = dudxdx[:, 1:-1, 1:-1]
    dudt = dudt[:, :, 1:-1]
    dudtdx = dudtdx[:, :, 1:-1]
    return t, x, u, dudx, dudxdx, dudt, dudtdx

get_burger_data = partial(gen_data, pde=burger, init_fn=generate_random_ics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_ics', type=int, default=200)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--dt', type=float, default=0.002)
    parser.add_argument('--L', type=int, default=20)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--n_terms', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./data')
    parser.add_argument('--save_name', type=str, default='train')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t, x, u = get_burger_data(n_ics=args.n_ics, L=args.L, n=args.n, n_terms=args.n_terms, dt=args.dt, num_steps=args.num_steps)
    t, x, u, dudx, dudxdx, dudt, dudtdx = compute_derivative(t, x, u)

    t = torch.from_numpy(t).to(torch.float32)
    x = torch.from_numpy(x).to(torch.float32)
    u = torch.from_numpy(u).to(torch.float32)
    dudx = torch.from_numpy(dudx).to(torch.float32)
    dudxdx = torch.from_numpy(dudxdx).to(torch.float32)
    dudt = torch.from_numpy(dudt).to(torch.float32)
    dudtdx = torch.from_numpy(dudtdx).to(torch.float32)
    torch.save(t, f'{args.save_dir}/burger/{args.save_name}-t.pt')
    torch.save(x, f'{args.save_dir}/burger/{args.save_name}-x.pt')
    torch.save(u, f'{args.save_dir}/burger/{args.save_name}-u.pt')
    torch.save(dudx, f'{args.save_dir}/burger/{args.save_name}-dudx.pt')
    torch.save(dudxdx, f'{args.save_dir}/burger/{args.save_name}-dudxdx.pt')
    torch.save(dudt, f'{args.save_dir}/burger/{args.save_name}-dudt.pt')
    torch.save(dudtdx, f'{args.save_dir}/burger/{args.save_name}-dudtdx.pt')