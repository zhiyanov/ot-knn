import numpy as np
import ot_estimators as ote
import sys
import time
import ot
import os

vocab = None
dataset = None
queries = None
answers = None
dataset_modified = None
queries_modified = None
dataset_weights = None
queries_weights = None
solver = None
scores = None
dataset_prep = None
dataset_dn = None
dataset_cap = None
queries_prep = None
queries_qn = None
qt = None
query_cap = None
point_prep = None
point_pn = None
pt = None
point_cap = None



def load_data(data_folder):
    global vocab
    global dataset
    global queries
    global answers
    global dataset_weights
    global queries_weights
    global dataset_modified
    global queries_modified
    global solver
    global scores
    global dataset_prep
    global dataset_dn
    global dataset_cap

    vocab = np.load(os.path.join(data_folder, 'vocab.npy'))
    vocab = vocab.astype(np.float32)
    dataset = np.load(os.path.join(data_folder, 'dataset.npy'), allow_pickle=True)
    queries = np.load(os.path.join(data_folder, 'queries.npy'), allow_pickle=True)
    dataset_weights = np.load(os.path.join(data_folder, 'dataset_weights.npy'), allow_pickle=True)
    queries_weights = np.load(os.path.join(data_folder, 'queries_weights.npy'), allow_pickle=True)
    answers = np.load(os.path.join(data_folder, 'answers.npy'))

    queries = queries[:answers.shape[1]]
    queries_weights = queries_weights[:answers.shape[1]]

    dataset_modified = [[(j, dataset_weights[i][j]) for j in range(len(dataset[i]))] for i in range(len(dataset))]
    queries_modified = [[(j, queries_weights[i][j]) for j in range(len(queries[i]))] for i in range(len(queries))]

    solver = ote.OTEstimators()
    solver.load_vocabulary(vocab)

    scores = np.zeros(len(dataset), dtype=np.float32)

    dataset_prep = []
    dataset_dn = []
    dataset_cap = []


    for i in range(len(dataset)):
        cur = []
        for j in dataset[i]:
            cur.append(vocab[j])
        cur = np.vstack(cur)
        dataset_prep.append(cur)
        dataset_dn.append(np.linalg.norm(cur, axis=1).reshape(-1, 1)**2)
        dataset_cap.append(np.array([dataset_weights[i][j] for j in range(len(dataset[i]))], dtype=np.float64))

def load_query(query, weights):
    global queries_prep
    global queries_qn
    global qt
    global query_cap
    queries_prep = []
    for j in query:
        queries_prep.append(vocab[j])
    queries_prep = np.vstack(queries_prep)
    queries_qn = np.linalg.norm(queries_prep, axis=1).reshape(1, -1)**2
    qt = np.transpose(queries_prep)
    query_cap = np.array([weights[j] for j in range(len(query))], dtype=np.float64)


def get_distance_matrix_dataset(first_point_id, second_point_id):
    dm = dataset_dn[first_point_id] \
         + dataset_dn[second_point_id].reshape(1, -1) \
         - 2.0 * np.dot(dataset_prep[first_point_id],
                        np.transpose(dataset_prep[second_point_id]))
    dm[dm < 0.0] = 0.0
    dm = np.sqrt(dm)
    return dm

def exact_emd_dataset(first_point_id, second_point_id):
    best = 1e100
    best_id = -1

    dm = get_distance_matrix_dataset(first_point_id, second_point_id)
    dm = dm.astype(np.float64)
    emd_score = ot.lp.emd2(dataset_cap[first_point_id],
                           dataset_cap[second_point_id], dm)
    return emd_score

def flowtree_dataset(first_point_id, second_point_id):
    return solver.flowtree_query(dataset_modified[first_point_id],
                                 dataset_modified[second_point_id])

def quadtree_dataset(first_point_id, second_point_id):
    return solver.quadtree_query(dataset_modified[first_point_id],
                                 dataset_modified[second_point_id])

def rwmd_dataset(first_point_id, second_point_id):
    dm = get_distance_matrix_dataset(first_point_id, second_point_id)
    score = max(np.mean(dm.min(axis=0)), np.mean(dm.min(axis=1)))
    return score

def lc_wmd_cost_dataset(dist, k):
    if dist.shape[0] > dist.shape[1]:
        dist = dist.T
    s1 = dist.shape[0]
    s2 = dist.shape[1]
    cost1 = np.mean(dist.min(axis=0))
    if s1 == s2:
        cost2 = np.mean(dist.min(axis=1))
        return max(cost1, cost2)
    k = min(k, int(np.floor(s2/s1)), s2-1)
    remainder = (1./s1) - k*(1./s2)
    pdist = np.partition(dist, k, axis=1)
    cost2 = (np.sum(pdist[:,:k]) * 1./s2) + (np.sum(pdist[:,k]) * remainder)
    return max(cost1, cost2)

def lc_wmd_dataset(first_point_id, second_point_id, k_param=1):
    dm = get_distance_matrix_dataset(first_point_id, second_point_id)
    score = lc_wmd_cost_dataset(dm, k_param)
    return score

def sinkhorn_cost_dataset(dist, n_iter=1):
    eta = 30

    A = np.exp(-eta*dist/dist.max())

    c = np.ones((1, dist.shape[1]), dtype=np.float32) / dist.shape[1]
    left_cap = np.ones((dist.shape[0], 1), dtype=np.float32) / dist.shape[0]

    for iii in range(n_iter):
        A *= (left_cap/np.sum(A, axis=1, keepdims=True))
        A *= (c/np.sum(A, axis=0, keepdims=True))
    x = left_cap/A.sum(1, keepdims=True)
    x[x>1] = 1.
    A = x*A
    y = c/A.sum(0, keepdims=True)
    y[y>1] = 1.
    A *= y
    err_r = (left_cap-A.sum(1, keepdims=True))
    err_r_t = (left_cap-A.sum(1, keepdims=True)).transpose()
    err_c = (c-A.sum(0, keepdims=True))
    A += np.matmul(err_r, err_c) /(np.abs(err_r_t)).sum()

    cost = (A*dist).sum()
    return cost

def sinkhorn_dataset(first_point_id, second_point_id, n_iter=1):
    dm = get_distance_matrix_dataset(first_point_id, second_point_id)
    score = sinkhorn_cost_dataset(dm, n_iter)
    return score

def load_query_modified(query_modified):
    global queries_prep
    global queries_qn
    global qt
    global query_cap
    queries_prep = []
    for q in query_modified:
        queries_prep.append(vocab[q[0]])
    queries_prep = np.vstack(queries_prep)
    queries_qn = np.linalg.norm(queries_prep, axis=1).reshape(1, -1)**2
    qt = np.transpose(queries_prep)
    query_cap = np.array([query_modified[j][1] for j in range(len(query))], dtype=np.float64)


def get_distance_matrix(point_id):
    dm = dataset_dn[point_id] + queries_qn - 2.0 * np.dot(dataset_prep[point_id], qt)
    dm[dm < 0.0] = 0.0
    dm = np.sqrt(dm)
    return dm

def exact_emd(query_modified, point_id):
    load_query_modified(query_modified)
    best = 1e100
    best_id = -1
    dm = get_distance_matrix(point_id)
    dm = dm.astype(np.float64)
    emd_score = ot.lp.emd2(dataset_cap[point_id], query_cap, dm)
    return emd_score

def flowtree(query_modified, point_id):
    return solver.flowtree_query(query_modified, dataset_modified[point_id])

def quadtree(query_modified, point_id):
    return solver.quadtree_query(query_modified, dataset_modified[point_id])

def rwmd(query_modified, point_id):
    load_query(query_modified)

    dm = get_distance_matrix(point_id)
    score = max(np.mean(dm.min(axis=0)), np.mean(dm.min(axis=1)))
    return score

def lc_wmd_cost(dist, k):
    if dist.shape[0] > dist.shape[1]:
        dist = dist.T
    s1 = dist.shape[0]
    s2 = dist.shape[1]
    cost1 = np.mean(dist.min(axis=0))
    if s1 == s2:
        cost2 = np.mean(dist.min(axis=1))
        return max(cost1, cost2)
    k = min(k, int(np.floor(s2/s1)), s2-1)
    remainder = (1./s1) - k*(1./s2)
    pdist = np.partition(dist, k, axis=1)
    cost2 = (np.sum(pdist[:,:k]) * 1./s2) + (np.sum(pdist[:,k]) * remainder)
    return max(cost1, cost2)

def lc_wmd(query_modified, point_id, k_param=1):
    load_query(query, query_weights)

    dm = get_distance_matrix(point_id)
    score = lc_wmd_cost_metric(dm, k_param)
    return score

def sinkhorn_cost(dist, n_iter=1):
    eta = 30

    A = np.exp(-eta*dist/dist.max())

    c = np.ones((1, dist.shape[1]), dtype=np.float32) / dist.shape[1]
    left_cap = np.ones((dist.shape[0], 1), dtype=np.float32) / dist.shape[0]

    for iii in range(n_iter):
        A *= (left_cap/np.sum(A, axis=1, keepdims=True))
        A *= (c/np.sum(A, axis=0, keepdims=True))
    x = left_cap/A.sum(1, keepdims=True)
    x[x>1] = 1.
    A = x*A
    y = c/A.sum(0, keepdims=True)
    y[y>1] = 1.
    A *= y
    err_r = (left_cap-A.sum(1, keepdims=True))
    err_r_t = (left_cap-A.sum(1, keepdims=True)).transpose()
    err_c = (c-A.sum(0, keepdims=True))
    A += np.matmul(err_r, err_c) /(np.abs(err_r_t)).sum()

    cost = (A*dist).sum()
    return cost

def sinkhorn(query_modified, point_id, n_iter=1):
    load_query(query, query_weights)

    dm = get_distance_matrix(point_id)
    score = sinkhorn_cost_metric(dm, n_iter)
    return score

def load_point(point, weights):
    global point_prep
    global point_pn
    global pt
    global point_cap
    point_prep = []
    for j in point:
        point_prep.append(vocab[j])
    point_prep = np.vstack(point_prep)
    point_pn = np.linalg.norm(point_prep, axis=1).reshape(-1, 1)**2
    pt = np.transpose(point_prep)
    point_cap = np.array([weights[j] for j in range(len(point))], dtype=np.float64)


def get_distance_matrix_metric():
    dm = point_pn + queries_qn - 2.0 * np.dot(point_prep, qt)
    dm[dm < 0.0] = 0.0
    dm = np.sqrt(dm)
    return dm

def exact_emd_metric(query, query_weights, point, point_weights):
    load_query(query, query_weights)
    load_point(point, point_weights)

    best = 1e100
    best_id = -1
    dm = get_distance_matrix_metric()
    dm = dm.astype(np.float64)
    emd_score = ot.lp.emd2(point_cap, query_cap, dm)
    return emd_score

def flowtree_metric(query, query_weights, point, point_weights):
    query_modified = [(query[j], query_weights[j]) for j in range(len(query))]
    point_modified = [(point[j], point_weights[j]) for j in range(len(point))]
    return solver.flowtree_query(query_modified, point_modified)

def quadtree_metric(query, query_weights, point, point_weights):
    query_modified = [(query[j], query_weights[j]) for j in range(len(query))]
    point_modified = [(point[j], point_weights[j]) for j in range(len(point))]
    return solve.quadtree_query(query_modified, point_modified)

def rwmd_metric(query, query_weights, point, point_weights):
    load_query(query, query_weights)
    load_point(point, point_weights)

    dm = get_distance_matrix_metric()
    score = max(np.mean(dm.min(axis=0)), np.mean(dm.min(axis=1)))
    return score

def lc_wmd_cost_metric(dist, k):
    if dist.shape[0] > dist.shape[1]:
        dist = dist.T
    s1 = dist.shape[0]
    s2 = dist.shape[1]
    cost1 = np.mean(dist.min(axis=0))
    if s1 == s2:
        cost2 = np.mean(dist.min(axis=1))
        return max(cost1, cost2)
    k = min(k, int(np.floor(s2/s1)), s2-1)
    remainder = (1./s1) - k*(1./s2)
    pdist = np.partition(dist, k, axis=1)
    cost2 = (np.sum(pdist[:,:k]) * 1./s2) + (np.sum(pdist[:,k]) * remainder)
    return max(cost1, cost2)

def lc_wmd_metric(query, query_weights, point, point_weights, k_param=1):
    load_query(query, query_weights)
    load_point(point, point_weights)

    dm = get_distance_matrix_metric()
    score = lc_wmd_cost_metric(dm, k_param)
    return score

def sinkhorn_cost_metric(dist, n_iter=1):
    eta = 30

    A = np.exp(-eta*dist/dist.max())

    c = np.ones((1, dist.shape[1]), dtype=np.float32) / dist.shape[1]
    left_cap = np.ones((dist.shape[0], 1), dtype=np.float32) / dist.shape[0]

    for iii in range(n_iter):
        A *= (left_cap/np.sum(A, axis=1, keepdims=True))
        A *= (c/np.sum(A, axis=0, keepdims=True))
    x = left_cap/A.sum(1, keepdims=True)
    x[x>1] = 1.
    A = x*A
    y = c/A.sum(0, keepdims=True)
    y[y>1] = 1.
    A *= y
    err_r = (left_cap-A.sum(1, keepdims=True))
    err_r_t = (left_cap-A.sum(1, keepdims=True)).transpose()
    err_c = (c-A.sum(0, keepdims=True))
    A += np.matmul(err_r, err_c) /(np.abs(err_r_t)).sum()

    cost = (A*dist).sum()
    return cost

def sinkhorn_metric(query, query_weights, point, point_weights, n_iter=1):
    load_query(query, query_weights)
    load_point(point, point_weights)

    dm = get_distance_matrix_metric()
    score = sinkhorn_cost_metric(dm, n_iter)
    return score
