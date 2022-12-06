from itertools import combinations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import samplers


def projection_distance(table, dim=5):
    is_noncontextual, hyperplane_idx = check_noncontexuality(table)
    if is_noncontextual:
        return 0
    coefs = list()
    for i in get_coefs(dim)[np.argmax(hyperplane_idx)]:
        coefs.extend(np.array([1, -1, -1, 1])*i)
    box = np.reshape(np.transpose(table), (1, -1))[0]
    return np.abs(np.dot(box, coefs) - dim + 2)/np.sqrt(np.sum(np.square(coefs)))


def sample(n: int):
    table = np.random.uniform(0, 1, (4, n))
    table[:, 0] /= np.sum(table[:, 0])
    for idx in range(n-2):
        next_idx = (idx + 1) % n
        table[[0, 1], next_idx] = table[[0, 1], next_idx] * np.sum(table[[0, 2], idx]) / np.sum(table[[0, 1], next_idx])
        table[[2, 3], next_idx] = table[[2, 3], next_idx] * np.sum(table[[1, 3], idx]) / np.sum(table[[2, 3], next_idx])
    table[0, n-1] = np.random.rand() * min(table[0, 0]+table[1, 0], table[0, n-2]+table[2, n-2])
    table[1, n-1] = table[0, n-2] + table[2, n-2] - table[0, n-1]
    table[2, n-1] = table[0, 0] + table[2, 0] - table[0, n-1]
    table[3, n-1] = 1 - np.sum(table[0:3, n-1])
    return table


def sampler(num: int = 10000, dim: int = 5, noncontextual: bool = True):
    samples = list()
    while len(samples) < num:
        s = sample(dim)
        if noncontextual == check_noncontexuality(s)[0]:
            samples.append(s)
            if len(samples) % 1000 == 0:
                print(len(samples))
    return samples


def check_consistency(box):
    col_dim = box.shape[1]
    results = list()
    for i in range(col_dim):
        condition_one = np.sum(box[[0,2], i]) - np.sum(box[[0,1], (i + 1)%col_dim])
        condition_two = np.sum(box[[1,3], i]) - np.sum(box[[2,3], (i + 1)%col_dim])
        results.append(condition_two==0 and condition_one==0)
        print(condition_two,condition_one)
    return np.all(results)

def get_coefs(n):
    for i in np.arange(1, n + 2, 2):
        comb = list(combinations(np.arange(n), i))
        buf = np.ones((len(comb), n))
        for idx, c in enumerate(comb):
            buf[idx, c] *= -1
        coefs = buf if not 'coefs' in locals() else np.vstack((coefs, buf))
    return coefs


def E(table):
    return np.tile(table[0, :] + table[3, :] - table[1, :] - table[2, :], (2 ** (table.shape[1] - 1), 1))


def check_noncontexuality(table):
    n = table.shape[1]
    results = np.sum(np.multiply(E(table), get_coefs(n)), axis=1)
    # print(results.shape)
    return np.all(results <= n - 2), results

def NL(behaviour, noncontextual, num=5000):
    return np.min(1 / 10 * np.sum(np.reshape(np.abs(behaviour - noncontextual), (int(num), -1)), axis=1))


def NL2(behaviour):
    return 1 / 10 * np.sum(np.abs(behaviour - 1 / 4 * np.ones((4, 5))))


def NL_square(behaviour, noncontextual, num=5000):
    return 1/10 * np.min(np.sqrt(np.sum(np.abs(behaviour - noncontextual) ** 2, axis=(1, 2))))


def get_dataloaders_c(num: int, batch_size: int = 500, fun=projection_distance):
    contextual_test, _ = samplers.prepare_mixed_states_from_10D_saved(num, 0, train=False)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(contextual_test).float(),
        torch.from_numpy(np.array(list(map(fun, contextual_test)))).long()
    )
    return DataLoader(dataset, batch_size)

def get_dataloaders_nc(num: int, batch_size: int = 50):
    _, noncontextual_test = samplers.prepare_mixed_states_from_10D_saved(0, num, train=False)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(noncontextual_test).float(),
        torch.from_numpy(np.array(list(map(projection_distance, noncontextual_test)))).long()
    )
    return DataLoader(dataset, batch_size)

def get_dataloaders(num: int, batch_size: int = 500):
    num = int(num // 2)
    contextual, noncontextual = samplers.prepare_mixed_states_from_10D_saved(num, num)

    data_X = np.concatenate((noncontextual, contextual))
    distances = np.zeros(len(contextual))
    m1 = np.concatenate((np.zeros(len(noncontextual)), distances))
    m2 = np.concatenate((np.zeros(len(noncontextual)), np.ones(len(distances))))
    data_y = np.transpose(np.vstack((m1, m2)))

    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y)
    ds1 = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train[:, 1]).long())
    ds2 = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test[:, 1]).long())
    ds3 = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train[:, 0]).long())
    ds4 = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test[:, 0]).long())

    return (DataLoader(ds1, batch_size),
            DataLoader(ds2, batch_size),
            DataLoader(ds3, batch_size),
            DataLoader(ds4, batch_size))



