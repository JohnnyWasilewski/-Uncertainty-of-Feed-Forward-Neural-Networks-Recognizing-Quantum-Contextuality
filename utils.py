from pathlib import Path
from scipy.stats import pearsonr
import numpy as np

import matplotlib.pyplot as plt

import uncertainties as unc
import dataloaders as dl
import models as m


def experiment(num, idx, device, metrics, results_path, batch_size=200, save_fig=True, test_num=500):
    data_loader_train, data_loader_test, data_loader_train_r, data_loader_test_r = dl.get_dataloaders(num, batch_size)
    contextual_data_loader = dl.get_dataloaders_c(test_num, batch_size)
    print('jestem!')
    classifier = m.Classifier(data_loader_train, data_loader_test, device)
    accuracies = classifier.train_and_eval()
    preds = classifier.pred(contextual_data_loader)
    dropout_var, dropout_mean, dropout_raw_var = unc.get_preds_dropout(contextual_data_loader, classifier.model, device)
    laplace_var, laplace_mean = unc.get_preds_laplace(data_loader_train, contextual_data_loader, classifier.model, device)

    print('policzyłem niepewności!')

    dists = list()
    for X, y in contextual_data_loader:
        dists.extend([dl.projection_distance(x.numpy()) for x in X])
    #metrics[(num, idx, "acc", "raw")] = classifier.acc
    #metrics[(num, idx, "acc_adjusted", "raw")] = acc
    print('Policzyłem dystanse!')
    def entropy(x):
        return x*np.log2(x) + (1-x)*np.log2(1-x)
    values = [preds, entropy(preds), laplace_mean, entropy(laplace_mean), dropout_mean, entropy(dropout_mean), laplace_var, dropout_var, dropout_raw_var]
    print_corr_results(pearsonr, dists, values, accuracies, num, idx, metrics, save_fig, results_path)
    #return dropout_var, dropout_mean, laplace, dists, acc, acc_raw


def print_corr_results(correlation_metric, preds_regressor, preds_uncertainty,  accuracies, num, idx, corr, save_fig, folder_name='NL_square3'):
    #corr[(num, idx, "preds", "raw")] = np.round(correlation_metric(preds_regressor, preds_uncertainty[0])[0], 2)
    #for preds, name in zip(preds_uncertainty[1:], ["L_var", "L_mean", "MCDO_var", "MCDO_mean"]):
        #preds_adjusted = list(map(lambda x: 4*(x-0.5)**2, preds))
        #corr[(num, idx, name, "raw")] = np.round(correlation_metric(preds_regressor, preds)[0], 2)
        #corr[(num, idx, name, "adjusted")] = np.round(correlation_metric(preds_regressor, preds_adjusted)[0], 2)
        #corr[(num, idx, name, "square")] = np.round(correlation_metric(preds_regressor, np.power(preds_adjusted, 2))[0], 2)
        #corr[(num, idx, name, "root")] = np.round(correlation_metric(preds_regressor, np.sqrt(preds))[0] - correlation_metric(preds_regressor, preds)[0], 2)

    row = [
        (num, idx),
        *[correlation_metric(preds_regressor, preds_uncertainty[i])[0] for i in range(9)],
        np.mean(preds_uncertainty[1]),
        np.mean(preds_uncertainty[3]),
        np.mean(preds_uncertainty[5]),
        np.mean(preds_uncertainty[6]),
        np.mean(preds_uncertainty[7]),
        np.mean(preds_uncertainty[8]),
        *accuracies,
        ]
    corr.loc[corr.shape[0]] = row
    if idx == 0:
        Path(f'{folder_name}/experiment_{num}').mkdir(parents=True, exist_ok=True)
    # fig, ax = plt.subplots(6, 1, clear=True, figsize=(7, 7))
    # for i, pred in enumerate(preds_uncertainty):
    #     ax[i].plot(preds_regressor, pred, 'p')
    # fig.suptitle('{} training points'.format(num), fontsize=16)
    # if save_fig:
    #     fig.savefig(f'{folder_name}/experiment_{num}/samples_{num}_{idx}.png')