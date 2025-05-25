import torch
import numpy as np
from adan import Adan
import os

save_model_path = './saved_models'

def train_model(model, train_loader, test_loader, task, opt, num_epochs, lr, device, log_interval, save_interval, save_dir, classify=False, **kwargs):
    if opt == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr)
    elif opt == "Adan":
        optimizer = Adan(model.parameters(), lr)
    if classify:
        loss_fn = torch.nn.BCELoss()
    else:
        loss_fn = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        running_losses = []
        model.train()
        for data in train_loader:
            if task == 'top':
                feature = data['x'].reshape(data['x'].shape[0], -1)
                label = data['y'].reshape(data['y'].shape[0], -1)
            elif task in ['heat', 'burger']:
                feature = torch.stack([data['u'], data['dudx'], data['dudxdx']], dim=1)
                label = data['dudt'].unsqueeze(1)
            elif task == 'kdv':
                feature = torch.stack([data['u'], data['dudx'], data['dudxdx'], data['dudxdxdx']], dim=1)
                label = data['dudt'].unsqueeze(1)
            elif task == 'wave':
                feature = torch.stack([data['u'], data['dudx'], data['dudy'], data['dudxdx'], data['dudydy'], data['dudxdy']], dim=1)
                label = data['dudtdt'].unsqueeze(1)
            elif task in ['rd', 'schrodinger']:
                feature = torch.stack([data['u'], data['dudx'], data['dudy'], data['dudxdx'], data['dudydy'], data['dudxdy'], data['v'], data['dvdx'], data['dvdy'], data['dvdxdx'], data['dvdydy'], data['dvdxdy']], dim=1)
                label = torch.stack([data['dudt'], data['dvdt']], dim=1)

            feature = feature.to(device)
            label = label.to(device)

            label_hat = model(feature)
            loss = loss_fn(label_hat, label)
            running_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % log_interval == 0:
            print(f'Epoch {epoch}, loss: {np.mean(running_losses):.4f}')
            model.eval()
            with torch.no_grad():
                running_losses = []
                for data in test_loader:
                    if task == 'top':
                        feature = data['x'].reshape(data['x'].shape[0], -1)
                        label = data['y'].reshape(data['y'].shape[0], -1)
                    elif task in ['heat', 'burger']:
                        feature = torch.stack([data['u'], data['dudx'], data['dudxdx']], dim=1)
                        label = data['dudt'].unsqueeze(1)
                    elif task == 'kdv':
                        feature = torch.stack([data['u'], data['dudx'], data['dudxdx'], data['dudxdxdx']], dim=1)
                        label = data['dudt'].unsqueeze(1)
                    elif task == 'wave':
                        feature = torch.stack([data['u'], data['dudx'], data['dudy'], data['dudxdx'], data['dudydy'], data['dudxdy']], dim=1)
                        label = data['dudtdt'].unsqueeze(1)
                    elif task in ['rd', 'schrodinger']:
                        feature = torch.stack([data['u'], data['dudx'], data['dudy'], data['dudxdx'], data['dudydy'], data['dudxdy'], data['v'], data['dvdx'], data['dvdy'], data['dvdxdx'], data['dvdydy'], data['dvdxdy']], dim=1)
                        label = torch.stack([data['dudt'], data['dvdt']], dim=1)

                    feature = feature.to(device)
                    label = label.to(device)
                    label_hat = model(feature)
                    loss = loss_fn(label_hat, label)
                    running_losses.append(loss.item())

                print(f'Epoch {epoch} test, loss: {np.mean(running_losses):.4f}')

        if (epoch + 1) % save_interval == 0:
            if not os.path.exists(f'{save_model_path}/{save_dir}'):
                os.makedirs(f'{save_model_path}/{save_dir}')
            torch.save(model.state_dict(), f'{save_model_path}/{save_dir}/model_{epoch}.pt')
    