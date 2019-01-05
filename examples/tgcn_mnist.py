"""A multi-layer perceptron for classification of MNIST handwritten digits."""
from __future__ import absolute_import, division
from __future__ import print_function

import autograd.numpy as np
import gcn.coarsening as coarsening
import gcn.graph as graph
from autograd import grad
from autograd.misc import flatten
from autograd.misc.optimizers import adam
from autograd.scipy.misc import logsumexp
from ..load.data import load_mnist


def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)


def accuracy_tgcn(params, inputs, targets):
    target_class    = np.argmax(targets, axis=1)
    predicted_class = np.argmax(nn_predict_tgcn_cheb(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)


def log_posterior_tgcn(params, inputs, targets, L2_reg):
    log_prior = L2_reg * l2_norm(params)
    log_lik = np.sum(nn_predict_tgcn_cheb(params, inputs) * targets)
    # return log_lik
    return log_prior + log_lik


def ReLU(x):
    return x * (x > 0)


def nn_predict_tgcn_cheb(params, x):

    L = graph.rescale_L(hyper['L'][0], lmax=2)
    w = np.fft.fft(x, axis=2)
    xc = chebyshev_time_vertex(L, w, hyper['K'])
    y = np.einsum('knhq,kfh->fnq', xc, params['W1'])
    y += np.expand_dims(params['b1'], axis=2)

    # nonlinear layer
    y = np.tanh(y)
    # y = ReLU(y)

    # dense layer
    y = np.einsum('fnq,cfn->cq', y, params['W2'])
    y += np.expand_dims(params['b2'], axis=1)

    outputs = np.real(y.T)
    return outputs - logsumexp(outputs, axis=1, keepdims=True)


def create_sq_mesh(m, n):
    # adjacency matrix
    A = np.zeros((m * n, m * n))
    for j in range(n):
        for i in range(m):
            # node id
            k = (j - 1) * m + i
            # edge north
            if i > 1:
                A[k, k - 1] = 1
            # edge south
            if i < m:
                A[k, k + 1] = 1
            # edge west
            if j > 1:
                A[k, k - m] = 1
            # edge east
            if j < n:
                A[k, k + m] = 1
    return A


def init_tgcn_params_coarsen_cheb(L, H):

    _, U = graph.fourier(L[0])

    hyper = dict()
    hyper['NFEATURES'] = U.shape[0]
    hyper['NCLASSES'] = 10
    hyper['N'] = L[0].shape[0]
    hyper['F'] = 15
    hyper['K'] = 10
    hyper['U'] = U
    hyper['L'] = L
    hyper['H'] = H

    params = dict()

    params['W1'] = 1.*np.random.randn(hyper['K'], hyper['F'], hyper['H'])
    params['b1'] = 1.*np.random.randn(hyper['F'], hyper['N'])

    params['W2'] = 1.*np.random.randn(hyper['NCLASSES'], hyper['F'], hyper['NFEATURES'])
    params['b2'] = 1.*np.random.randn(hyper['NCLASSES'])

    return params, hyper


def create_graph():

    def grid_graph(m, corners=False):
        z = graph.grid(m)
        dist, idx = graph.distance_sklearn_metrics(z, k=number_edges, metric=metric)
        A = graph.adjacency(dist, idx)

        if corners:
            import scipy.sparse
            A = A.toarray()
            A[A < A.max() / 1.5] = 0
            A = scipy.sparse.csr_matrix(A)
            print('{} edges'.format(A.nnz))

        print("{} > {} edges".format(A.nnz // 2, number_edges * m ** 2 // 2))
        return A

    number_edges= 12
    metric = 'euclidean'
    normalized_laplacian = True
    coarsening_levels = 4

    A = grid_graph(28, corners=False)
    graphs, perm = coarsening.coarsen(A, levels=coarsening_levels, self_connections=False)
    L = [graph.laplacian(A, normalized=normalized_laplacian) for A in graphs]
    del A

    return L, perm


def chebyshev_time_vertex(L, X, K):
    """Return T_k X where T_k are the Chebyshev polynomials of order up to K.
    Complexity is O(KMN)."""
    X = np.transpose(X, axes=[1, 2, 0])
    M, N, Q = X.shape
    Xt = np.empty((K, M, N, Q), dtype='complex')
    # Xt[0, ...] = X
    # if K > 1:
    #     Xt[1, ...] = L.dot(X)
    # for k in range(2, K):
    #     Xt[k, ...] = 2 * L.dot(Xt[k-1, ...]) - Xt[k-2, ...]
    for q in range(Q):
        Xt[0, :, :, q] = X[:, :, q]
        if K > 1:
            Xt[1, :, :, q] = L.dot(X[:, :, q])
        for k in range(2, K):
            Xt[k, :, :, q] = 2 * L.dot(Xt[k-1, :, :, q]) - Xt[k-2, :, :, q]

    return Xt


def get_mnist_time_data_autograd(perm):

    N, train_data, train_labels, test_data, test_labels = load_mnist()

    H = 12

    train_data = np.transpose(np.tile(train_data, (H, 1, 1)), axes=[1, 2, 0])
    test_data = np.transpose(np.tile(test_data, (H, 1, 1)), axes=[1, 2, 0])

    idx_train = range(2*512)
    idx_test = range(2*512)

    train_data = train_data[idx_train]
    train_labels = train_labels[idx_train]
    test_data = test_data[idx_test]
    test_labels = test_labels[idx_test]

    train_data = perm_data_time(train_data, perm)
    test_data = perm_data_time(test_data, perm)

    del perm

    return train_data, test_data, train_labels, test_labels


def perm_data_time(x, indices):
    """
    Permute data matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return x

    N, M, Q = x.shape
    Mnew = len(indices)
    assert Mnew >= M
    xnew = np.empty((N, Mnew, Q))
    for i,j in enumerate(indices):
        # Existing vertex, i.e. real data.
        if j < M:
            xnew[:, i, :] = x[:, j, :]
        # Fake vertex because of singeltons.
        # They will stay 0 so that max pooling chooses the singelton.
        # Or -infty ?
        else:
            xnew[:, i, :] = np.zeros((N, Q))
    return xnew

# #################################################

global hyper

if __name__ == '__main__':

    batch_size = 256
    num_epochs = 500
    step_size = 0.005
    L2_reg = 0.0
    param_scale = 0.1

    L, perm = create_graph()
    # train_images, test_images, train_labels, test_labels = get_mnist_data_autograd(perm)
    train_images, test_images, train_labels, test_labels = get_mnist_time_data_autograd(perm)

    num_batches = int(np.ceil(len(train_images) / batch_size))

    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)


    # init_params_gcn, hyper = init_gcn_params_coarsen_cheb(L)
    init_params_tgcn, hyper = init_tgcn_params_coarsen_cheb(L, train_images.shape[2])


    def print_perf_tgcn(params, iter, gradient):
        if iter % num_batches == 0:
            train_acc = accuracy_tgcn(params, train_images, train_labels)
            test_acc  = accuracy_tgcn(params, test_images, test_labels)
            print("{:15}|{:20.6}|{:20.6}".format(iter//num_batches, train_acc, test_acc))

            flattened, _ = flatten(gradient)
            ng = np.dot(flattened, flattened)
            print('{:1.3e}'.format(ng))


    def objective_tgcn(params, iter):
        idx = batch_indices(iter)
        return -log_posterior_tgcn(params, train_images[idx], train_labels[idx], L2_reg)

    # Get gradient of objective using autograd.
    objective_tgcn_grad = grad(objective_tgcn)

    optimized_params = adam(objective_tgcn_grad, init_params_tgcn, step_size=step_size,
                            num_iters=num_epochs * num_batches, callback=print_perf_tgcn)
