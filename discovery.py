import torch
from torch.autograd.functional import jvp

def compute_theta(data, task, device):
    if task == 'top':
        theta = {}
        theta['th'] = data['x'][0]
    elif task in ['heat', 'burger']:
        t, x, u, dudx, dudxdx, dudt = data['t'], data['x'], data['u'], data['dudx'], data['dudxdx'], data['dudt']
        theta = {}
        theta['th'] = torch.tensor([1, t, x, u, t ** 2, x ** 2, u ** 2, t * x, t * u, x * u], device=device)
        theta['dth_dx'] = torch.tensor([0, 0, 1, dudx, 0, 2 * x, 2 * u * dudx, t, t * dudx, u + x * dudx], device=device)
        theta['dth_dxdx'] = torch.tensor([0, 0, 0, dudxdx, 0, 2, 2 * dudx ** 2 + 2 * u * dudxdx, 0, t * dudxdx, 2 * dudx + x * dudxdx], device=device)
        theta['dth_dt'] = torch.tensor([0, 1, 0, dudt, 2 * t, 0, 2 * u * dudt, x, u + t * dudt, x * dudt], device=device)
    elif task == 'kdv':
        t, x, u, dudx, dudxdx, dudxdxdx, dudt = data['t'], data['x'], data['u'], data['dudx'], data['dudxdx'], data['dudxdxdx'], data['dudt']
        theta = {}
        theta['th'] = torch.tensor([1, t, x, u, t ** 2, x ** 2, u ** 2, t * x, t * u, x * u], device=device)
        theta['dth_dx'] = torch.tensor([0, 0, 1, dudx, 0, 2 * x, 2 * u * dudx, t, t * dudx, u + x * dudx], device=device)
        theta['dth_dxdx'] = torch.tensor([0, 0, 0, dudxdx, 0, 2, 2 * dudx ** 2 + 2 * u * dudxdx, 0, t * dudxdx, 2 * dudx + x * dudxdx], device=device)
        theta['dth_dxdxdx'] = torch.tensor([0, 0, 0, dudxdxdx, 0, 0, 6 * dudx * dudxdx + 2 * u * dudxdxdx, 0, t * dudxdxdx, 3 * dudxdx + x * dudxdxdx], device=device)
        theta['dth_dt'] = torch.tensor([0, 1, 0, dudt, 2 * t, 0, 2 * u * dudt, x, u + t * dudt, x * dudt], device=device)
    elif task == 'wave':
        t, x, y, u, dudt, dudx, dudy, dudtdt, dudxdx, dudydy, dudxdy = data['t'], data['x'], data['y'], data['u'], data['dudt'], data['dudx'], data['dudy'], data['dudtdt'], data['dudxdx'], data['dudydy'], data['dudxdy']
        theta = {}
        theta['th'] = torch.tensor([1, t, x, y, u, t ** 2, x ** 2, y ** 2, u ** 2, t * x, t * y, t * u, x * y, x * u, y * u], device=device)
        theta['dth_dt'] = torch.tensor([0, 1, 0, 0, dudt, 2 * t, 0, 0, 2 * u * dudt, x, y, u + t * dudt, 0, x * dudt, y * dudt], device=device)
        theta['dth_dx'] = torch.tensor([0, 0, 1, 0, dudx, 0, 2 * x, 0, 2 * u * dudx, t, 0, t * dudx, y, u + x * dudx, y * dudx], device=device)
        theta['dth_dy'] = torch.tensor([0, 0, 0, 1, dudy, 0, 0, 2 * y, 2 * u * dudy, 0, t, t * dudy, x, x * dudy, u + y * dudy], device=device)
        theta['dth_dtdt'] = torch.tensor([0, 0, 0, 0, dudtdt, 2, 0, 0, 2 * (dudt ** 2 + u * dudtdt), 0, 0, 2 * dudt + t * dudtdt, 0, x * dudtdt, y * dudtdt], device=device)
        theta['dth_dxdx'] = torch.tensor([0, 0, 0, 0, dudxdx, 0, 2, 0, 2 * (dudx ** 2 + u * dudxdx), 0, 0, t * dudxdx, 0, 2 * dudx + x * dudxdx, y * dudxdx], device=device)
        theta['dth_dydy'] = torch.tensor([0, 0, 0, 0, dudydy, 0, 0, 2, 2 * (dudy ** 2 + u * dudydy), 0, 0, t * dudydy, 0, x * dudydy, 2 * dudy + y * dudydy], device=device)
        theta['dth_dxdy'] = torch.tensor([0, 0, 0, 0, dudxdy, 0, 0, 0, 2 * (dudx * dudy + u * dudxdy), 0, 0, t * dudxdy, 1, dudy + x * dudxdy, dudx + y * dudxdy], device=device)
    elif task in ['rd', 'schrodinger']:
        t, x, y = data['t'], data['x'], data['y']
        u, dudt, dudx, dudy, dudxdx, dudydy, dudxdy = data['u'], data['dudt'], data['dudx'], data['dudy'], data['dudxdx'], data['dudydy'], data['dudxdy']
        v, dvdt, dvdx, dvdy, dvdxdx, dvdydy, dvdxdy = data['v'], data['dvdt'], data['dvdx'], data['dvdy'], data['dvdxdx'], data['dvdydy'], data['dvdxdy']
        theta = {}
        theta['th'] = torch.tensor([1, t, x, y, u, v, t ** 2, x ** 2, y ** 2, u ** 2, v ** 2, t * x, t * y, t * u, t * v, x * y, x * u, x * v, y * u, y * v, u * v], device=device)
        theta['dth_dt'] = torch.tensor([0, 1, 0, 0, dudt, dvdt, 2 * t, 0, 0, 2 * u * dudt, 2 * v * dvdt, x, y, u + t * dudt, v + t * dvdt, 0, x * dudt, x * dvdt, y * dudt, y * dvdt, dudt * v + u * dvdt], device=device)
        theta['dth_dx'] = torch.tensor([0, 0, 1, 0, dudx, dvdx, 0, 2 * x, 0, 2 * u * dudx, 2 * v * dvdx, t, 0, t * dudx, t * dvdx, y, u + x * dudx, v + x * dvdx, y * dudx, y * dvdx, dudx * v + u * dvdx], device=device)
        theta['dth_dy'] = torch.tensor([0, 0, 0, 1, dudy, dvdy, 0, 0, 2 * y, 2 * u * dudy, 2 * v * dvdy, 0, t, t * dudy, t * dvdy, x, x * dudy, x * dvdy, u + y * dudy, v + y * dvdy, dudy * v + u * dvdy], device=device)
        theta['dth_dxdx'] = torch.tensor([0, 0, 0, 0, dudxdx, dvdxdx, 0, 2, 0, 2 * (dudx ** 2 + u * dudxdx), 2 * (dvdx ** 2 + v * dvdxdx), 0, 0, t * dudxdx, t * dvdxdx, 0, 2 * dudx + x * dudxdx, 2 * dvdx + x * dvdxdx, y * dudxdx, y * dvdxdx, dudxdx * v + 2 * dudx * dvdx + u * dvdxdx], device=device)
        theta['dth_dydy'] = torch.tensor([0, 0, 0, 0, dudydy, dvdydy, 0, 0, 2, 2 * (dudy ** 2 + u * dudydy), 2 * (dvdy ** 2 + v * dvdydy), 0, 0, t * dudydy, t * dvdydy, 0, x * dudydy, x * dvdydy, 2 * dudy + y * dudydy, 2 * dvdy + y * dvdydy, dudydy * v + 2 * dudy * dvdy + u * dvdydy], device=device)
        theta['dth_dxdy'] = torch.tensor([0, 0, 0, 0, dudxdy, dvdxdy, 0, 0, 0, 2 * (dudx * dudy + u * dudxdy), 2 * (dvdx * dvdy + v * dvdxdy), 0, 0, t * dudxdy, t * dvdxdy, 1, dudy + x * dudxdy, dvdy + x * dvdxdy, dudx + y * dudxdy, dvdx + y * dvdxdy, dudxdy * v + dudx * dvdy + dudy * dvdx + u * dvdxdy], device=device)
    return theta

def compute_theta_n(data, task, device):
    theta = compute_theta(data, task, device)
    if task == 'top':
        th = theta['th']
        theta_n = torch.zeros((th.shape[0] * 4, 16), device=device)
        for i in range(th.shape[0]):
            for j in range(4):
                theta_n[4*i+j, 4*j:4*(j + 1)] = th[i]
    elif task in ['heat', 'burger']:
        t, x, u, dudx, dudxdx, dudt, dudtdx = data['t'], data['x'], data['u'], data['dudx'], data['dudxdx'], data['dudt'], data['dudtdx']
        th, dth_dx, dth_dxdx, dth_dt = theta['th'], theta['dth_dx'], theta['dth_dxdx'], theta['dth_dt']
        theta_n = torch.zeros((4, 30), device=device)
        theta_n[0, 20:] = th
        theta_n[1, :10] = -dudt * dth_dx
        theta_n[1, 10:20] = -dudx * dth_dx
        theta_n[1, 20:] = dth_dx
        theta_n[2, :10] = -(dudt * dth_dxdx + 2 * dudtdx * dth_dx)
        theta_n[2, 10:20] = -(dudx * dth_dxdx + 2 * dudxdx * dth_dx)
        theta_n[2, 20:] = dth_dxdx
        theta_n[3, :10] = -dudt * dth_dt
        theta_n[3, 10:20] = -dudx * dth_dt
        theta_n[3, 20:] = dth_dt
    elif task == 'kdv':
        t, x, u, dudx, dudxdx, dudxdxdx, dudt, dudtdx, dudtdxdx = data['t'], data['x'], data['u'], data['dudx'], data['dudxdx'], data['dudxdxdx'], data['dudt'], data['dudtdx'], data['dudtdxdx']
        th, dth_dx, dth_dxdx, dth_dxdxdx, dth_dt = theta['th'], theta['dth_dx'], theta['dth_dxdx'], theta['dth_dxdxdx'], theta['dth_dt']
        theta_n = torch.zeros((5, 30), device=device)
        theta_n[0, 20:] = th
        theta_n[1, :10] = -dudt * dth_dx
        theta_n[1, 10:20] = -dudx * dth_dx
        theta_n[1, 20:] = dth_dx
        theta_n[2, :10] = -(dudt * dth_dxdx + 2 * dudtdx * dth_dx)
        theta_n[2, 10:20] = -(dudx * dth_dxdx + 2 * dudxdx * dth_dx)
        theta_n[2, 20:] = dth_dxdx
        theta_n[3, :10] = -(dudt * dth_dxdxdx + 3 * dudtdx * dth_dxdx + 3 * dudtdxdx * dth_dx)
        theta_n[3, 10:20] = -(dudx * dth_dxdxdx + 3 * dudxdx * dth_dxdx + 3 * dudxdxdx * dth_dx)
        theta_n[3, 20:] = dth_dxdxdx
        theta_n[4, :10] = -dudt * dth_dt
        theta_n[4, 10:20] = -dudx * dth_dt
        theta_n[4, 20:] = dth_dt
    elif task == 'wave':
        t, x, y, u, dudt, dudx, dudy, dudtdt, dudxdx, dudydy, dudtdx, dudtdy, dudxdy = data['t'], data['x'], data['y'], data['u'], data['dudt'], data['dudx'], data['dudy'], data['dudtdt'], data['dudxdx'], data['dudydy'], data['dudtdx'], data['dudtdy'], data['dudxdy']
        th, dth_dt, dth_dx, dth_dy, dth_dtdt, dth_dxdx, dth_dydy, dth_dxdy = theta['th'], theta['dth_dt'], theta['dth_dx'], theta['dth_dy'], theta['dth_dtdt'], theta['dth_dxdx'], theta['dth_dydy'], theta['dth_dxdy']
        theta_n = torch.zeros((7, 60), device=device)
        theta_n[0, 45:] = th
        theta_n[1, :15] = -dudt * dth_dx
        theta_n[1, 15:30] = -dudx * dth_dx
        theta_n[1, 30:45] = -dudy * dth_dx
        theta_n[1, 45:] = dth_dx
        theta_n[2, :15] = -dudt * dth_dy
        theta_n[2, 15:30] = -dudx * dth_dy
        theta_n[2, 30:45] = -dudy * dth_dy
        theta_n[2, 45:] = dth_dy
        theta_n[3, :15] = -(dudt * dth_dxdx + 2 * dudtdx * dth_dx)
        theta_n[3, 15:30] = -(dudx * dth_dxdx + 2 * dudxdx * dth_dx)
        theta_n[3, 30:45] = -(dudy * dth_dxdx + 2 * dudxdy * dth_dx)
        theta_n[3, 45:] = dth_dxdx
        theta_n[4, :15] = -(dudt * dth_dydy + 2 * dudtdy * dth_dy)
        theta_n[4, 15:30] = -(dudx * dth_dydy + 2 * dudxdy * dth_dy)
        theta_n[4, 30:45] = -(dudy * dth_dydy + 2 * dudydy * dth_dy)
        theta_n[4, 45:] = dth_dydy
        theta_n[5, :15] = -(dudt * dth_dxdy + dudtdx * dth_dy + dudtdy * dth_dx)
        theta_n[5, 15:30] = -(dudx * dth_dxdy + dudxdx * dth_dy + dudxdy * dth_dx)
        theta_n[5, 30:45] = -(dudy * dth_dxdy + dudxdy * dth_dy + dudydy * dth_dx)
        theta_n[5, 45:] = dth_dxdy
        theta_n[6, :15] = -(dudt * dth_dtdt + 2 * dudtdt * dth_dt)
        theta_n[6, 15:30] = -(dudx * dth_dtdt + 2 * dudtdx * dth_dt)
        theta_n[6, 30:45] = -(dudy * dth_dtdt + 2 * dudtdy * dth_dt)
        theta_n[6, 45:] = dth_dtdt
    elif task in ['rd', 'schrodinger']:
        t, x, y = data['t'], data['x'], data['y']
        u, dudt, dudx, dudy, dudxdx, dudydy, dudtdx, dudtdy, dudxdy = data['u'], data['dudt'], data['dudx'], data['dudy'], data['dudxdx'], data['dudydy'], data['dudtdx'], data['dudtdy'], data['dudxdy']
        v, dvdt, dvdx, dvdy, dvdxdx, dvdydy, dvdtdx, dvdtdy, dvdxdy = data['v'], data['dvdt'], data['dvdx'], data['dvdy'], data['dvdxdx'], data['dvdydy'], data['dvdtdx'], data['dvdtdy'], data['dvdxdy']
        th, dth_dt, dth_dx, dth_dy, dth_dxdx, dth_dydy, dth_dxdy = theta['th'], theta['dth_dt'], theta['dth_dx'], theta['dth_dy'], theta['dth_dxdx'], theta['dth_dydy'], theta['dth_dxdy']
        theta_n = torch.zeros((14, 105), device=device)
        theta_n[0, 63:84] = th
        theta_n[1, :21] = -dudt * dth_dx
        theta_n[1, 21:42] = -dudx * dth_dx
        theta_n[1, 42:63] = -dudy * dth_dx
        theta_n[1, 63:84] = dth_dx
        theta_n[2, :21] = -dudt * dth_dy
        theta_n[2, 21:42] = -dudx * dth_dy
        theta_n[2, 42:63] = -dudy * dth_dy
        theta_n[2, 63:84] = dth_dy
        theta_n[3, :21] = -(dudt * dth_dxdx + 2 * dudtdx * dth_dx)
        theta_n[3, 21:42] = -(dudx * dth_dxdx + 2 * dudxdx * dth_dx)
        theta_n[3, 42:63] = -(dudy * dth_dxdx + 2 * dudxdy * dth_dx)
        theta_n[3, 63:84] = dth_dxdx
        theta_n[4, :21] = -(dudt * dth_dydy + 2 * dudtdy * dth_dy)
        theta_n[4, 21:42] = -(dudx * dth_dydy + 2 * dudxdy * dth_dy)
        theta_n[4, 42:63] = -(dudy * dth_dydy + 2 * dudydy * dth_dy)
        theta_n[4, 63:84] = dth_dydy
        theta_n[5, :21] = -(dudt * dth_dxdy + dudtdx * dth_dy + dudtdy * dth_dx)
        theta_n[5, 21:42] = -(dudx * dth_dxdy + dudxdx * dth_dy + dudxdy * dth_dx)
        theta_n[5, 42:63] = -(dudy * dth_dxdy + dudxdy * dth_dy + dudydy * dth_dx)
        theta_n[5, 63:84] = dth_dxdy
        theta_n[6, 84:] = th
        theta_n[7, :21] = -dvdt * dth_dx
        theta_n[7, 21:42] = -dvdx * dth_dx
        theta_n[7, 42:63] = -dvdy * dth_dx
        theta_n[7, 84:] = dth_dx
        theta_n[8, :21] = -dvdt * dth_dy
        theta_n[8, 21:42] = -dvdx * dth_dy
        theta_n[8, 42:63] = -dvdy * dth_dy
        theta_n[8, 84:] = dth_dy
        theta_n[9, :21] = -(dvdt * dth_dxdx + 2 * dvdtdx * dth_dx)
        theta_n[9, 21:42] = -(dvdx * dth_dxdx + 2 * dvdxdx * dth_dx)
        theta_n[9, 42:63] = -(dvdy * dth_dxdx + 2 * dvdxdy * dth_dx)
        theta_n[9, 84:] = dth_dxdx
        theta_n[10, :21] = -(dvdt * dth_dydy + 2 * dvdtdy * dth_dy)
        theta_n[10, 21:42] = -(dvdx * dth_dydy + 2 * dvdxdy * dth_dy)
        theta_n[10, 42:63] = -(dvdy * dth_dydy + 2 * dvdydy * dth_dy)
        theta_n[10, 84:] = dth_dydy
        theta_n[11, :21] = -(dvdt * dth_dxdy + dvdtdx * dth_dy + dvdtdy * dth_dx)
        theta_n[11, 21:42] = -(dvdx * dth_dxdy + dvdxdx * dth_dy + dvdxdy * dth_dx)
        theta_n[11, 42:63] = -(dvdy * dth_dxdy + dvdxdy * dth_dy + dvdydy * dth_dx)
        theta_n[11, 84:] = dth_dxdy
        theta_n[12, :21] = -dudt * dth_dt
        theta_n[12, 21:42] = -dudx * dth_dt
        theta_n[12, 42:63] = -dudy * dth_dt
        theta_n[12, 63:84] = dth_dt
        theta_n[13, :21] = -dvdt * dth_dt
        theta_n[13, 21:42] = -dvdx * dth_dt
        theta_n[13, 42:63] = -dvdy * dth_dt
        theta_n[13, 84:] = dth_dt
    return theta_n

def symmetry_discovery(model, train_loader, task, device, threshold, sample, **kwargs):
    model.eval()
    with torch.no_grad():
        if task == 'top':
            CTC = torch.zeros((16, 16), device=device)
        elif task in ['heat', 'burger', 'kdv']:
            CTC = torch.zeros((30, 30), device=device)
        elif task == 'wave':
            CTC = torch.zeros((60, 60), device=device)
        elif task in ['rd', 'schrodinger']:
            CTC = torch.zeros((105, 105), device=device)
        for j, data in enumerate(train_loader):
            if j < sample * 100:
                continue
            elif j >= (sample + 1) * 100:
                break
            for key in data:
                data[key] = data[key].to(device)
            if task == 'top':
                feature = data['x'].reshape(data['x'].shape[0], -1)
            elif task in ['heat', 'burger']:
                feature = torch.stack([data['u'], data['dudx'], data['dudxdx']], dim=1)
            elif task == 'kdv':
                feature = torch.stack([data['u'], data['dudx'], data['dudxdx'], data['dudxdxdx']], dim=1)
            elif task == 'wave':
                feature = torch.stack([data['u'], data['dudx'], data['dudy'], data['dudxdx'], data['dudydy'], data['dudxdy']], dim=1)
            elif task in ['rd', 'schrodinger']:
                feature = torch.stack([data['u'], data['dudx'], data['dudy'], data['dudxdx'], data['dudydy'], data['dudxdy'], data['v'], data['dvdx'], data['dvdy'], data['dvdxdx'], data['dvdydy'], data['dvdxdy']], dim=1)
            theta_n = compute_theta_n(data, task, device)
            if task == 'top':
                df = torch.zeros((kwargs['output_dim'], kwargs['input_dim']), device=device)
            if task in ['heat', 'burger', 'kdv', 'wave', 'rd', 'schrodinger']:
                df = torch.zeros((kwargs['output_dim'], kwargs['input_dim'] + kwargs['output_dim']), device=device)
                df[:, kwargs['input_dim']:] = -torch.eye(kwargs['output_dim'], device=device)
            for i in range(kwargs['input_dim']):
                v = torch.zeros(kwargs['input_dim'], device=device)
                v[i] = 1
                df[:, i] = jvp(model, feature[0], v=v)[1]
            Ci = df @ theta_n
            CTC += Ci.T @ Ci
        _, S2, Vh = torch.linalg.svd(CTC.to(torch.float64))
        S2, Vh = S2.to(torch.float32), Vh.to(torch.float32)
        S = torch.sqrt(S2)
        return S, Vh[S < threshold]
