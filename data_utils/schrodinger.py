import numpy as np
import torch
from tqdm import trange
from functools import partial
import argparse

def generate_random_ics(n_ics=200, L=20, n=100, n_terms=10):
    dx = L / (n + 1)
    x2 = np.linspace(-L / 2, L / 2, n + 1)
    x = x2[:n]
    y = x
    x, y = np.meshgrid(x, y)
    initial_conditions_u = []
    for _ in range(n_ics):
        a0 = np.random.randn()
        u0 = a0 / 4
        for m in range(1, n_terms + 1):
            for n in range(1, n_terms + 1):
                amn = np.random.randn()
                bmn = np.random.randn()
                cmn = np.random.randn()
                dmn = np.random.randn()
                u0 += amn * np.cos(2 * m * np.pi * x / L) * np.cos(2 * n * np.pi * y / L)
                u0 += bmn * np.cos(2 * m * np.pi * x / L) * np.sin(2 * n * np.pi * y / L)
                u0 += cmn * np.sin(2 * m * np.pi * x / L) * np.cos(2 * n * np.pi * y / L)
                u0 += dmn * np.sin(2 * m * np.pi * x / L) * np.sin(2 * n * np.pi * y / L)
        initial_conditions_u.append(u0)
    initial_conditions_v = []
    for _ in range(n_ics):
        a0 = np.random.randn()
        v0 = a0 / 4
        for m in range(1, n_terms + 1):
            for n in range(1, n_terms + 1):
                amn = np.random.randn()
                bmn = np.random.randn()
                cmn = np.random.randn()
                dmn = np.random.randn()
                v0 += amn * np.cos(2 * m * np.pi * x / L) * np.cos(2 * n * np.pi * y / L)
                v0 += bmn * np.cos(2 * m * np.pi * x / L) * np.sin(2 * n * np.pi * y / L)
                v0 += cmn * np.sin(2 * m * np.pi * x / L) * np.cos(2 * n * np.pi * y / L)
                v0 += dmn * np.sin(2 * m * np.pi * x / L) * np.sin(2 * n * np.pi * y / L)
        initial_conditions_v.append(v0)
    return np.array(initial_conditions_u), np.array(initial_conditions_v), x, y, dx

def schrodinger(u, dudxdx, dudydy, v, dvdxdx, dvdydy):
    dudt = -0.5 * (dvdydy + dvdxdx) + v * u ** 2 + v ** 3
    dvdt = 0.5 * (dudxdx + dudydy) - u * v ** 2 - u ** 3
    return dudt, dvdt

def space_derivative(u, dx):
    dudx = (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / (2 * dx)
    dudy = (np.roll(u, -1, axis=-2) - np.roll(u, 1, axis=-2)) / (2 * dx)
    dudxdx = (np.roll(u, -1, axis=-1) - 2 * u + np.roll(u, 1, axis=-1)) / (dx ** 2)
    dudydy = (np.roll(u, -1, axis=-2) - 2 * u + np.roll(u, 1, axis=-2)) / (dx ** 2)
    dudxdy = (np.roll(np.roll(u, -1, axis=-2), -1, axis=-1) + np.roll(np.roll(u, 1, axis=-2), 1, axis=-1) - np.roll(np.roll(u, -1, axis=-2), 1, axis=-1) - np.roll(np.roll(u, 1, axis=-2), -1, axis=-1)) / (4 * dx ** 2)
    return dudx, dudy, dudxdx, dudydy, dudxdy

def gen_data(pde, init_fn, n_ics=200, L=20, n=100, n_terms=10, dt=0.002, num_steps=1000):
    t = np.arange(0, dt * num_steps, dt)
    u0, v0, x, y, dx = init_fn(n_ics=n_ics, L=L, n=n, n_terms=n_terms)
    u = np.zeros((num_steps, *u0.shape))
    v = np.zeros_like(u)
    u[0] = u0
    v[0] = v0
    for i in trange(num_steps):
        u1 = u[i]
        v1 = v[i]
        dudx1, dudy1, dudxdx1, dudydy1, dudxdy1 = space_derivative(u1, dx)
        dvdx1, dvdy1, dvdxdx1, dvdydy1, dvdxdy1 = space_derivative(v1, dx)
        dudt1, dvdt1 = pde(u1, dudxdx1, dudydy1, v1, dvdxdx1, dvdydy1)
        if i == num_steps - 1:
            break
        k1u = dt * dudt1
        k1v = dt * dvdt1
        u2 = u[i] + 0.5 * k1u
        v2 = v[i] + 0.5 * k1v
        dudx2, dudy2, dudxdx2, dudydy2, dudxdy2 = space_derivative(u2, dx)
        dvdx2, dvdy2, dvdxdx2, dvdydy2, dvdxdy2 = space_derivative(v2, dx)
        dudt2, dvdt2 = pde(u2, dudxdx2, dudydy2, v2, dvdxdx2, dvdydy2)
        k2u = dt * dudt2
        k2v = dt * dvdt2
        u3 = u[i] + 0.5 * k2u
        v3 = v[i] + 0.5 * k2v
        dudx3, dudy3, dudxdx3, dudydy3, dudxdy3 = space_derivative(u3, dx)
        dvdx3, dvdy3, dvdxdx3, dvdydy3, dvdxdy3 = space_derivative(v3, dx)
        dudt3, dvdt3 = pde(u3, dudxdx3, dudydy3, v3, dvdxdx3, dvdydy3)
        k3u = dt * dudt3
        k3v = dt * dvdt3
        u4 = u[i] + k3u
        v4 = v[i] + k3v
        dudx4, dudy4, dudxdx4, dudydy4, dudxdy4 = space_derivative(u4, dx)
        dvdx4, dvdy4, dvdxdx4, dvdydy4, dvdxdy4 = space_derivative(v4, dx)
        dudt4, dvdt4 = pde(u4, dudxdx4, dudydy4, v4, dvdxdx4, dvdydy4)
        k4u = dt * dudt4
        k4v = dt * dvdt4
        u[i + 1] = u[i] + (k1u + 2 * k2u + 2 * k3u + k4u) / 6
        v[i + 1] = v[i] + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    u = np.transpose(u, (1, 0, 2, 3))
    v = np.transpose(v, (1, 0, 2, 3))
    return t, x, y, u, v

def compute_derivative(t, x, y, u, v):
    dt = t[1] - t[0]
    dx = x[0, 1] - x[0, 0]
    dudx, dudy, dudxdx, dudydy, dudxdy = space_derivative(u, dx)
    dudt = (u[:, 2:, :, :] - u[:, :-2, :, :]) / (2 * dt)
    dudtdx, dudtdy, _, _, _ = space_derivative(dudt, dx)
    dvdx, dvdy, dvdxdx, dvdydy, dvdxdy = space_derivative(v, dx)
    dvdt = (v[:, 2:, :, :] - v[:, :-2, :, :]) / (2 * dt)
    dvdtdx, dvdtdy, _, _, _ = space_derivative(dvdt, dx)
    t = t[1:-1]
    x = x[1:-1:10, 1:-1:10]
    y = y[1:-1:10, 1:-1:10]
    u = u[:, 1:-1, 1:-1:10, 1:-1:10]
    dudt = dudt[:, :, 1:-1:10, 1:-1:10]
    dudx = dudx[:, 1:-1, 1:-1:10, 1:-1:10]
    dudy = dudy[:, 1:-1, 1:-1:10, 1:-1:10]
    dudxdx = dudxdx[:, 1:-1, 1:-1:10, 1:-1:10]
    dudydy = dudydy[:, 1:-1, 1:-1:10, 1:-1:10]
    dudtdx = dudtdx[:, :, 1:-1:10, 1:-1:10]
    dudtdy = dudtdy[:, :, 1:-1:10, 1:-1:10]
    dudxdy = dudxdy[:, 1:-1, 1:-1:10, 1:-1:10]
    v = v[:, 1:-1, 1:-1:10, 1:-1:10]
    dvdt = dvdt[:, :, 1:-1:10, 1:-1:10]
    dvdx = dvdx[:, 1:-1, 1:-1:10, 1:-1:10]
    dvdy = dvdy[:, 1:-1, 1:-1:10, 1:-1:10]
    dvdxdx = dvdxdx[:, 1:-1, 1:-1:10, 1:-1:10]
    dvdydy = dvdydy[:, 1:-1, 1:-1:10, 1:-1:10]
    dvdtdx = dvdtdx[:, :, 1:-1:10, 1:-1:10]
    dvdtdy = dvdtdy[:, :, 1:-1:10, 1:-1:10]
    dvdxdy = dvdxdy[:, 1:-1, 1:-1:10, 1:-1:10]
    return t, x, y, u, dudt, dudx, dudy, dudxdx, dudydy, dudtdx, dudtdy, dudxdy, v, dvdt, dvdx, dvdy, dvdxdx, dvdydy, dvdtdx, dvdtdy, dvdxdy

get_schrodinger_data = partial(gen_data, pde=schrodinger, init_fn=generate_random_ics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_ics', type=int, default=200)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--dt', type=float, default=0.002)
    parser.add_argument('--L', type=int, default=20)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--n_terms', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default='./data')
    parser.add_argument('--save_name', type=str, default='train')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t, x, y, u, v = get_schrodinger_data(n_ics=args.n_ics, L=args.L, n=args.n, n_terms=args.n_terms, dt=args.dt, num_steps=args.num_steps)
    t, x, y, u, dudt, dudx, dudy, dudxdx, dudydy, dudtdx, dudtdy, dudxdy, v, dvdt, dvdx, dvdy, dvdxdx, dvdydy, dvdtdx, dvdtdy, dvdxdy = compute_derivative(t, x, y, u, v)

    t = torch.from_numpy(t).to(torch.float32)
    x = torch.from_numpy(x).to(torch.float32)
    y = torch.from_numpy(y).to(torch.float32)
    u = torch.from_numpy(u).to(torch.float32)
    dudt = torch.from_numpy(dudt).to(torch.float32)
    dudx = torch.from_numpy(dudx).to(torch.float32)
    dudy = torch.from_numpy(dudy).to(torch.float32)
    dudxdx = torch.from_numpy(dudxdx).to(torch.float32)
    dudydy = torch.from_numpy(dudydy).to(torch.float32)
    dudtdx = torch.from_numpy(dudtdx).to(torch.float32)
    dudtdy = torch.from_numpy(dudtdy).to(torch.float32)
    dudxdy = torch.from_numpy(dudxdy).to(torch.float32)
    v = torch.from_numpy(v).to(torch.float32)
    dvdt = torch.from_numpy(dvdt).to(torch.float32)
    dvdx = torch.from_numpy(dvdx).to(torch.float32)
    dvdy = torch.from_numpy(dvdy).to(torch.float32)
    dvdxdx = torch.from_numpy(dvdxdx).to(torch.float32)
    dvdydy = torch.from_numpy(dvdydy).to(torch.float32)
    dvdtdx = torch.from_numpy(dvdtdx).to(torch.float32)
    dvdtdy = torch.from_numpy(dvdtdy).to(torch.float32)
    dvdxdy = torch.from_numpy(dvdxdy).to(torch.float32)
    torch.save(t, f'{args.save_dir}/schrodinger/{args.save_name}-t.pt')
    torch.save(x, f'{args.save_dir}/schrodinger/{args.save_name}-x.pt')
    torch.save(y, f'{args.save_dir}/schrodinger/{args.save_name}-y.pt')
    torch.save(u, f'{args.save_dir}/schrodinger/{args.save_name}-u.pt')
    torch.save(dudt, f'{args.save_dir}/schrodinger/{args.save_name}-dudt.pt')
    torch.save(dudx, f'{args.save_dir}/schrodinger/{args.save_name}-dudx.pt')
    torch.save(dudy, f'{args.save_dir}/schrodinger/{args.save_name}-dudy.pt')
    torch.save(dudxdx, f'{args.save_dir}/schrodinger/{args.save_name}-dudxdx.pt')
    torch.save(dudydy, f'{args.save_dir}/schrodinger/{args.save_name}-dudydy.pt')
    torch.save(dudtdx, f'{args.save_dir}/schrodinger/{args.save_name}-dudtdx.pt')
    torch.save(dudtdy, f'{args.save_dir}/schrodinger/{args.save_name}-dudtdy.pt')
    torch.save(dudxdy, f'{args.save_dir}/schrodinger/{args.save_name}-dudxdy.pt')
    torch.save(v, f'{args.save_dir}/schrodinger/{args.save_name}-v.pt')
    torch.save(dvdt, f'{args.save_dir}/schrodinger/{args.save_name}-dvdt.pt')
    torch.save(dvdx, f'{args.save_dir}/schrodinger/{args.save_name}-dvdx.pt')
    torch.save(dvdy, f'{args.save_dir}/schrodinger/{args.save_name}-dvdy.pt')
    torch.save(dvdxdx, f'{args.save_dir}/schrodinger/{args.save_name}-dvdxdx.pt')
    torch.save(dvdydy, f'{args.save_dir}/schrodinger/{args.save_name}-dvdydy.pt')
    torch.save(dvdtdx, f'{args.save_dir}/schrodinger/{args.save_name}-dvdtdx.pt')
    torch.save(dvdtdy, f'{args.save_dir}/schrodinger/{args.save_name}-dvdtdy.pt')
    torch.save(dvdxdy, f'{args.save_dir}/schrodinger/{args.save_name}-dvdxdy.pt')