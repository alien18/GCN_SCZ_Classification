import numpy as np
from scipy.spatial import distance
from scipy.sparse import coo_matrix, csr
import torch
from torch_geometric.utils import dense_to_sparse
from neurocombat_sklearn import CombatModel
from torch_geometric.data import InMemoryDataset, Data


def compute_KNN_graph(matrix, k_degree=10, metric='euclidean'):
    """ Calculate the adjacency matrix from the connectivity matrix."""

    dist = distance.pdist(matrix, metric)
    dist = distance.squareform(dist)

    idx = np.argsort(dist)[:, 1:k_degree + 1]
    dist.sort()
    dist = dist[:, 1:k_degree + 1]

    A = adjacency(dist, idx).astype(np.float32)

    return A


def adjacency(dist, idx):

    m, k = dist.shape
    assert m, k == idx.shape
    assert dist.min() >= 0

    # Weights.
    sigma2 = np.mean(dist[:, -1]) ** 2
    dist = np.exp(- dist ** 2 / sigma2)

    # Weight matrix.
    I = np.arange(0, m).repeat(k)
    J = idx.reshape(m * k)
    V = dist.reshape(m * k)
    W = coo_matrix((V, (I, J)), shape=(m, m))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is csr.csr_matrix
    return W.todense()


def combat_trans(train_dataset, test_dataset, batch_train, batch_test, case_train, case_test):
    x_train = extract_fc_vals(train_dataset)
    x_test = extract_fc_vals(test_dataset)

    model = CombatModel()
    x_train_harmonized = model.fit_transform(x_train, batch_train, case_train)
    x_test_harmonized = model.transform(x_test, batch_test, case_test)
    restore_geometric_format(x_train_harmonized, train_dataset)
    restore_geometric_format(x_test_harmonized, test_dataset)

    return train_dataset, test_dataset


def extract_fc_vals(dataset):
    fc_vals = [distance.squareform(data.x.numpy()) for data in dataset]
    return np.array(fc_vals)


def restore_geometric_format(harmonized_data, dataset):

    data_list = []
    for i in range(harmonized_data.shape[0]):
        fc_mat = distance.squareform(harmonized_data[i])
        fc_mat = torch.from_numpy(fc_mat).float()
        adj = compute_KNN_graph(fc_mat)
        adj = torch.from_numpy(adj).float()
        edge_index, edge_attr = dense_to_sparse(adj)
        data = Data(x=fc_mat, edge_attr=edge_attr, edge_index=edge_index, y=dataset[i].y)
        data_list.append(data)
    dataset.data, dataset.slices = InMemoryDataset.collate(data_list)
    dataset.set_new_indices()
