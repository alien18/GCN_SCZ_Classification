import numpy as np
from scipy.spatial import distance
from scipy.sparse import coo_matrix, csr


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
