import torch
import torch.nn as nn

import numpy as np
import laplace

def entropy(x):
    return -1 * (x*np.log2(x + 0.000000000001) + (1-x)*np.log2(1-x + 0.000000000001))

# ------------------------------------------------------------
# ------------------------- LAPLACE --------------------------
# ------------------------------------------------------------

def get_preds_laplace(train_dataloader, predict_dataloader, model, device):
    model.to('cpu')
    la = laplace.Laplace(model, likelihood='classification')
    la.fit(train_dataloader)
    preds = list()
    for X, _ in predict_dataloader:
        preds.append(la.predictive_samples(x=X, n_samples=1000)[:, :, 1])
    p = torch.concat(preds, dim=1).numpy()
    return p

# -------------------------------------------------------------------------
# ------------------------------ DROPOUT ----------------------------------
# --------------------------------------------------------------------------
def get_preds_dropout(dataloader, model, device):
    model.eval()
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

    iter_num = 100
    y_true, y_preds, y_preds_raw = [], [], []
    softmax = nn.Softmax(dim=1)
    k,l=[],[]
    for X, y in dataloader:
        y_preds_tmp, y_preds_raw_tmp = [], []
        for _ in range(iter_num):
            X = X.to(device)
            y_hat = softmax(model(X))[:, 1].cpu().detach().numpy()
            l.append(softmax(model(X))[:, 0].cpu().detach().numpy())
            y_preds_tmp.append(y_hat)
            y_preds_raw_tmp.append(model(X)[:, 1].cpu().detach().numpy())
        y_true.append(y.numpy())
        y_preds.append(np.array(y_preds_tmp))
        y_preds_raw.append(np.array(y_preds_raw_tmp))
        k.append(np.array(l))
    raw = np.concatenate(y_preds_raw, axis=1)
    m = np.concatenate(y_preds, axis=1)
    unc = entropy(np.count_nonzero(m < 0.5, axis=0) / iter_num)
    unc_raw = entropy(np.count_nonzero(raw < 0.5, axis=0) / iter_num)
    return m


def calculate_uncertainties(arr: np.array):
    T = arr.shape[0]
    pred_sum = np.sum(arr, axis=0)
    entropy_uncertainty = entropy(pred_sum/T)
    mutual_info = entropy_uncertainty + 1/T * np.sum(entropy(arr), axis=0)
    var_ratio = 1 - np.count_nonzero(arr < 0.5, axis=0)/T

    aleatoric_uncertainty = np.mean(entropy(arr), axis=0)
    return {
        "entropy": entropy_uncertainty,
        "mutual_info": mutual_info,
        "var_ratio": var_ratio,
        "aleatoric_uncertainty": aleatoric_uncertainty
            }, [entropy_uncertainty, mutual_info, var_ratio, aleatoric_uncertainty]


import hamiltorch
def get_HMC_preds(
        train_data,
        net,
        test_data,
        device='cuda:0',
        num_samples=1000,
        tau=1,
        step_size=0.005,
        L=30,
        burn=100,
        store_on_GPU=False,
        debug=False,
        model_loss='multi_class_linear_output',
        mass=1.0,
):

    X, Y = train_data
    X_test, Y_test = test_data
    X = torch.FloatTensor(X)
    Y = torch.tensor(Y)


    # Effect of tau
    # Set to tau = 1000. to see a function that is less bendy (weights restricted to small bends)
    # Set to tau = 1. for more flexible
     # Prior Precision
    r = 0 # Random seed


    tau_list = []
    for w in net.parameters():
        tau_list.append(tau) # set the prior precision to be the same for each set of weights
    tau_list = torch.tensor(tau_list).to(device)

    # Set initial weights
    params_init = hamiltorch.util.flatten(net).to(device).clone()
    # Set the Inverse of the Mass matrix
    inv_mass = torch.ones(params_init.shape) / mass

    integrator = hamiltorch.Integrator.EXPLICIT
    sampler = hamiltorch.Sampler.HMC_NUTS

    hamiltorch.set_random_seed(r)
    params_hmc_f = hamiltorch.sample_model(net, X.to(device), Y.to(device), params_init=params_init,
                                           model_loss=model_loss, num_samples=num_samples,
                                           burn = burn, inv_mass=inv_mass.to(device),step_size=step_size,
                                           num_steps_per_sample=L, tau_list=tau_list,
                                           debug=debug, store_on_GPU=store_on_GPU,
                                           sampler = sampler)

    # At the moment, params_hmc_f is on the CPU so we move to GPU

    params_hmc_gpu = [ll.to(device) for ll in params_hmc_f[1:]]
    X_test = torch.FloatTensor(X_test)
    Y_test = torch.FloatTensor(Y_test)

    # Let's predict over the entire test range [-2,2]
    pred_list, log_probs_f = hamiltorch.predict_model(net,
                                                      x = X_test.to(device),
                                                      y = Y_test.to(device),
                                                      samples=params_hmc_gpu,
                                                      model_loss=model_loss,
                                                      tau_list=tau_list
                                                     )

    s = nn.Softmax(dim=2)
    return s(pred_list)[:,:, 0].cpu().numpy()