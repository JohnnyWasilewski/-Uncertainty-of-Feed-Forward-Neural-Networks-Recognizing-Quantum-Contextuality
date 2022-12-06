import os
import csv

import numpy as np
import pandas as pd

from utils import experiment


def run():
    def prepare_experiment_settings(num, vals):
        l = list()
        for v in vals:
            l.append(np.hstack((np.ones((num, 1)) * v, np.expand_dims(np.arange(num), 1))))
        return np.vstack(l)

    results_path = os.path.join("results", "dense_full_lowModelSD")
    os.makedirs(results_path,  exist_ok=True)
    metrics = pd.DataFrame(columns=
        ["id", "preds", "uncert", "L_mean", "L_uncert", "MCDO_mean", "MCDO_uncert", "L_var", "MCDO_var", "MCDO_var_raw",
         "mean_entropy", "mean_L_entropy", "mean_MCDO_entropy", "mean_L_var", "mean_MCDO_var", "mean_MCDO_raw_var", "acc_train", "acc_test"])

    args = prepare_experiment_settings(100, [5000, 10000])
    for num, idx in args:
        experiment(num, idx, 'cpu', metrics, results_path)
    metrics.to_csv(os.path.join(results_path, "results.csv"))


if __name__ == '__main__':
    run()
