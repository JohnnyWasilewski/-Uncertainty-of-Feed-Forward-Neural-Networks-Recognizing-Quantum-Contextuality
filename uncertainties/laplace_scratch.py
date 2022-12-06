import time

import torch
import torch.nn as nn

import numpy as np

# Energy function hessian calculation
def inverse_hessian(classifier, data_loader, device):
    criterion = nn.BCEWithLogitsLoss()
    print("Began hessian calculating")
    t0 = time.time()
    t = t0
    weights_num = np.sum([np.prod(v.shape) for v in classifier.parameters()])
    A = 0.01 * torch.eye(weights_num)
    for index, batch in enumerate(data_loader):
        for x, y in zip(batch[0], batch[1]):
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).unsqueeze(0).to(device)
            grads = torch.autograd.grad(criterion(classifier(x), y), classifier.parameters())
            flat = torch.cat([g.view(-1) for g in grads]).unsqueeze(1)
            A = _update_inverse_hessian(A, flat)
        if index % 10 == 0:
            t_ = time.time()
            time_diff = t_ - t
            t = t_
            print(
                f"Processed {(index + 1) * len(batch[0])} from {len(data_loader) * len(batch[0])} examples with norm {torch.norm(A)} in {time_diff}")
    print(f"Hessian calculated in {time.time() - t0}")
    return A


def _update_inverse_hessian(A, flat):
    a1 = A @ flat
    a2 = torch.t(flat) @ A
    a3 = a1 @ a2
    a4 = a2 @ flat + 1
    A = torch.sub(A, a3, alpha=1 / a4.item())
    return A


def compute_sigma(x, model, A):
    grads = torch.autograd.grad(model(x), model.parameters())
    flat = torch.cat([g.view(-1) for g in grads]).unsqueeze(1)
    sigma = torch.t(flat) @ A
    sigma = sigma @ flat
    return sigma, grads, flat


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def uncertainty_prediction(x, model_raw, model, inv_hessian):
    x = x.unsqueeze(0)
    var, grads, flat = compute_sigma(x, model, inv_hessian)
    ys = np.random.normal(model(x).item(), var.item(), size=100000)
    preds_mcmc = np.mean(sigmoid(ys))
    preds_boosted_mackay = torch.sigmoid(1 / np.sqrt(1 + np.pi * (var / 8)) * model(x))
    preds_raw = torch.sigmoid(model_raw(x))
    return preds_raw.item(), preds_boosted_mackay.item(), preds_mcmc.item(), var


def _get_preds_laplace(data_loader, model_raw, model, train_dataloader, device):
    model.eval()
    model.to(device)
    model_raw.to(device)
    preds_boosted_mackay, preds_raw, preds_mcmc, var = list(), list(), list(), list()
    inv_hessian = inverse_hessian(model, train_dataloader, device)
    print('Hessian calculated')
    t0 = time.time()
    t = t0
    for index, batch in enumerate(data_loader):
        for x in batch[0]:
            uncertainty = uncertainty_prediction(x, model_raw, model, inv_hessian)
            preds_raw.append(uncertainty[0])
            preds_boosted_mackay.append(uncertainty[1])
            preds_mcmc.append(uncertainty[2])
            var.append(uncertainty[3])
        t_ = time.time()
        time_diff = t_ - t
        t = t_
        print(f'Calculated {(index + 1) * len(batch[0])} samples from {len(data_loader) * len(batch[0])} in {time_diff}')
    print(f"Uncertainties calculated in {time.time() - t0}")
    return preds_raw, preds_boosted_mackay, preds_mcmc, var
