import itertools

import numpy as np

import dataloaders as dl


def get_vertices(n: int):
    contextual_edges = list()
    for v in filter(lambda x: sum(x) % 2 == 1, itertools.product([0, 1], repeat=n)):
        vertex = list()
        for ind in v:
            column = np.array([0, .5, .5, 0]) if ind == 1 else np.array([.5, 0, 0, .5])
            vertex.append(column)
        vertex = np.transpose(np.vstack(vertex))
        contextual_edges.append(vertex)

    candidates = list()
    for idxs in itertools.product([0, 1, 2, 3], repeat=n):
        tmp = np.zeros((4, n))
        for i in range(n):
            tmp[idxs[i], i] = 1
        candidates.append(tmp)
    noncontextual_edges = np.array(candidates)[list(map(dl.check_consistency, candidates))]

    return contextual_edges, noncontextual_edges


def _get_mixed_states(edges, num):
    mixed_states = list()
    m = len(edges)
    k = np.random.dirichlet(np.ones(len(edges)), size=int(num))

    for i in k:
        mixed_state = np.zeros(shape=edges[0].shape)
        for j in range(len(i)):
            mixed_state += (i[j]*edges[j])
        mixed_states.append(mixed_state)
    return mixed_states


def get_mixed_states(contextual_edges, noncontextual_edges, contextual_number, noncontextual_number):
    mixed_states_noncontextual = _get_mixed_states(noncontextual_edges, noncontextual_number)

    mixed_states_contextual = list()
    for contextual_edge in contextual_edges:
        distances = np.array([np.sum(np.square(contextual_edge - x)) for x in noncontextual_edges])
        edges = np.array(noncontextual_edges)[np.where(distances == np.min(distances))]
        edges = np.append(edges, np.expand_dims(contextual_edge, 0), axis=0)
        mixed_states_contextual.extend(*[_get_mixed_states(edges, contextual_number // len(contextual_edges))])
    return mixed_states_contextual, mixed_states_noncontextual


def prepare_mixed_states(dim:int, contextual_number, noncontextual_number):
    contextual_edges, noncontextual_edges = get_vertices(dim)
    return get_mixed_states(contextual_edges, noncontextual_edges, contextual_number, noncontextual_number)


def _check_positivity(vector, n=5):
    box = np.empty((4, n))
    for i in range(n):
        p_i1 = 1 + vector[i] + vector[(i+1) % n] + vector[n+i]
        p_i2 = 1 + vector[i] - vector[(i+1) % n] - vector[n+i]
        p_i3 = 1 - vector[i] + vector[(i+1) % n] - vector[n+i]
        p_i4 = 1 - vector[i] - vector[(i+1) % n] + vector[n+i]
        is_proper_column = p_i1 >= 0 and p_i2 >= 0 and p_i3 >= 0 and p_i4 >= 0
        if is_proper_column:
            box[:, i] = [p_i1 / 4, p_i2 / 4, p_i3 / 4, p_i4 / 4]
        else:
            return
    return box


def prepare_mixed_states_from_10D_saved(contextual_number, noncontextual_number, train=True, threshold=150000):
    contextual = np.load("data/contextual_all.npy")
    noncontextual = np.load("data/noncontextual_all.npy")
    noncontextual, contextual = (noncontextual[:threshold], contextual[:threshold]) if train else (noncontextual[threshold:], contextual[threshold:])
    return contextual[np.random.choice(len(contextual), contextual_number)], noncontextual[np.random.choice(len(noncontextual), noncontextual_number)]


def prepare_mixed_states_from_10D(dim: int, contextual_number, noncontextual_number):
    num = np.max((int(contextual_number * 100000 / 7), 100 * noncontextual_number))
    uniform = np.random.uniform(-1, 1,  (num, 2*dim))
    candidates = [_check_positivity(i, dim) for i in uniform]
    candidates = [l for l in candidates if l is not None]
    noncontextual, contextual = list(), list()
    for x in candidates:
        noncontextual.append(x) if dl.check_noncontexuality(x)[0] else contextual.append(x)
    return contextual, noncontextual[:noncontextual_number]
