"""A multi-layer perceptron for classification of MNIST handwritten digits."""
from __future__ import absolute_import, division
from __future__ import print_function
from autograd.scipy.misc import logsumexp
from data import load_mnist

# import data_mnist
import matplotlib.pyplot as plt
import scipy.sparse

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
import gcn.graph as graph
import gcn.coarsening as coarsening

from autograd import grad
from autograd.misc import flatten
from autograd.misc.optimizers import adam


def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]


def neural_net_predict(params, inputs):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs - logsumexp(outputs, axis=1, keepdims=True)


def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)


def log_posterior(params, inputs, targets, L2_reg):
    log_prior = -L2_reg * l2_norm(params)
    log_lik = np.sum(neural_net_predict(params, inputs) * targets)
    return log_prior + log_lik


def accuracy(params, inputs, targets):
    target_class    = np.argmax(targets, axis=1)
    predicted_class = np.argmax(neural_net_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)


def accuracy_GCN(params, inputs, targets):
    target_class    = np.argmax(targets, axis=1)
    predicted_class = np.argmax(nn_predict_GCN(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)


# ######################## GCN BLOCK ##########################

def ReLU(x):
    return x * (x > 0)


def dReLU(x):
    return 1. * (x > 0)


def log_posterior_GCN(params, inputs, targets, L2_reg):
    log_prior = -L2_reg * l2_norm(params)
    log_lik = np.sum(nn_predict_GCN(params, inputs) * targets)
    return log_prior + log_lik


def nn_predict_GCN(params, x):

    # x: NSAMPLES x NFEATURES
    U = hyper['U']
    xf = np.matmul(x, U)
    xf = np.expand_dims(xf, 1)  # NSAMPLES x 1 x NFEATURES
    xf = np.transpose(xf)  # NFEATURES x 1 x NSAMPLES

    # Filter
    yf = np.matmul(params['W1'], xf)  # for each feature
    yf = np.transpose(yf)  # NSAMPLES x NFILTERS x NFEATURES
    yf = np.reshape(yf, [-1, hyper['NFEATURES']])

    # Transform back to graph domain
    Ut = np.transpose(U)
    y = np.matmul(yf, Ut)
    y = np.reshape(y, [-1, hyper['F'], hyper['NFEATURES']])
    y += params['b1']  # NSAMPLES x NFILTERS x NFEATURES

    # nonlinear layer
    # y = ReLU(y)
    y = np.tanh(y)

    # dense layer
    y = np.reshape(y, [-1, hyper['F']*hyper['NFEATURES']])
    y = np.matmul(y, params['W2']) + params['b2']

    # # y = ReLU(y)
    # y = np.tanh(y)
    # y = np.matmul(y, params['W3']) + params['b3']

    return y


def create_sq_mesh(M, N):
    # adjacency matrix
    A = np.zeros((M * N, M * N))
    for j in range(N):
        for i in range(M):
            # node id
            k = (j - 1) * M + i
            # edge north
            if i > 1:
                A[k, k - 1] = 1
            # edge south
            if i < M:
                A[k, k + 1] = 1
            # edge west
            if j > 1:
                A[k, k - M] = 1
            # edge east
            if j < N:
                A[k, k + M] = 1
    return A


def init_GCN_params():

    A = scipy.sparse.csr_matrix(create_sq_mesh(28, 28))
    L = graph.laplacian(A)
    _, U = graph.fourier(L)

    hyper = dict()
    hyper['NFEATURES'] = 28**2
    hyper['NCLASSES'] = 10
    hyper['F'] = 15
    hyper['U'] = U

    params = dict()
    params['W1'] = np.random.randn(hyper['NFEATURES'], hyper['F'], 1)
    params['b1'] = np.random.randn(1, hyper['F'], 1)
    params['W2'] = np.random.randn(hyper['F']*hyper['NFEATURES'], hyper['NCLASSES'])
    params['b2'] = np.random.randn(hyper['NCLASSES'])

    # params['W2'] = np.random.randn(hyper['F']*hyper['NFEATURES'], 100)
    # params['b2'] = np.random.randn(100)
    # params['W3'] = np.random.randn(100, hyper['NCLASSES'])
    # params['b3'] = np.random.randn(hyper['NCLASSES'])

    return params, hyper


def init_GCN_params_coarsen(L):

    _, U = graph.fourier(L[0])

    hyper = dict()
    hyper['NFEATURES'] = U.shape[0]
    hyper['NCLASSES'] = 10
    hyper['F'] = 10
    hyper['U'] = U

    params = dict()
    params['W1'] = np.random.randn(hyper['NFEATURES'], hyper['F'], 1)
    params['b1'] = np.random.randn(1, hyper['F'], 1)
    params['W2'] = np.random.randn(hyper['F']*hyper['NFEATURES'], hyper['NCLASSES'])
    params['b2'] = np.random.randn(hyper['NCLASSES'])

    # params['W2'] = np.random.randn(hyper['F']*hyper['NFEATURES'], 100)
    # params['b2'] = np.random.randn(100)
    # params['W3'] = np.random.randn(100, hyper['NCLASSES'])
    # params['b3'] = np.random.randn(hyper['NCLASSES'])

    return params, hyper


# ####################################################################################

def create_graph():

    def grid_graph(m, corners=False):
        z = graph.grid(m)
        dist, idx = graph.distance_sklearn_metrics(z, k=number_edges, metric=metric)
        A = graph.adjacency(dist, idx)

        # Connections are only vertical or horizontal on the grid.
        # Corner vertices are connected to 2 neightbors only.
        if corners:
            import scipy.sparse
            A = A.toarray()
            A[A < A.max() / 1.5] = 0
            A = scipy.sparse.csr_matrix(A)
            print('{} edges'.format(A.nnz))

        print("{} > {} edges".format(A.nnz // 2, number_edges * m ** 2 // 2))
        return A


    number_edges= 8
    metric = 'euclidean'
    normalized_laplacian = True
    coarsening_levels = 4

    A = grid_graph(28, corners=False)
    # A = graph.replace_random_edges(A, 0)
    graphs, perm = coarsening.coarsen(A, levels=coarsening_levels, self_connections=False)
    L = [graph.laplacian(A, normalized=normalized_laplacian) for A in graphs]
    # graph.plot_spectrum(L)
    del A

    return L, perm


def get_MNIST_Data_Autograd(perm):

    N, train_data, train_labels, test_data, test_labels = load_mnist()

    train_data = coarsening.perm_data(train_data, perm)
    test_data = coarsening.perm_data(test_data, perm)

    del perm

    return train_data, test_data, train_labels, test_labels


# def get_MNIST_Data_Tf(perm):
#
#     import os
#     dir_data = os.path.join('..', 'data', 'mnist')
#
#     from tensorflow.examples.tutorials.mnist import input_data
#     mnist = input_data.read_data_sets(dir_data, one_hot=True)
#
#     train_data = mnist.train.images.astype(np.float32)
#     val_data = mnist.validation.images.astype(np.float32)
#     test_data = mnist.test.images.astype(np.float32)
#     train_labels = mnist.train.labels
#     val_labels = mnist.validation.labels
#     test_labels = mnist.test.labels
#
#     train_data = coarsening.perm_data(train_data, perm)
#     val_data = coarsening.perm_data(val_data, perm)
#     test_data = coarsening.perm_data(test_data, perm)
#     del perm
#
#     return train_data, test_data, train_labels, test_labels


global hyper

if __name__ == '__main__':

    batch_size = 256
    num_epochs = 25
    step_size = 0.001
    L2_reg = 1.0
    param_scale = 0.1

    # print("Loading training data...")

    L, perm = create_graph()
    # train_images1, test_images1, train_labels1, test_labels1 = get_MNIST_Data_Tf(perm)
    train_images, test_images, train_labels, test_labels = get_MNIST_Data_Autograd(perm)

    num_batches = int(np.ceil(len(train_images) / batch_size))

    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    print("     Epoch     |    Train accuracy  |       Test accuracy  ")
    def print_perf(params, iter, gradient):
        if iter % num_batches == 0:
            train_acc = accuracy(params, train_images, train_labels)
            test_acc  = accuracy(params, test_images, test_labels)
            print("{:15}|{:20}|{:20}".format(iter//num_batches, train_acc, test_acc))


    # ########### MLP ######################

    if False:

        # Model parameters
        layer_sizes = [784, 200, 100, 10]

        init_params = init_random_params(param_scale, layer_sizes)

        # Define training objective
        def objective(params, iter):
            idx = batch_indices(iter)
            return -log_posterior(params, train_images[idx], train_labels[idx], L2_reg)

        # Get gradient of objective using autograd.
        objective_grad = grad(objective)

        # The optimizers provided can optimize lists, tuples, or dicts of parameters.
        optimized_params = adam(objective_grad, init_params, step_size=step_size,
                                num_iters=num_epochs * num_batches, callback=print_perf)

    # ############### GCN #################

    # init_params_GCN, hyper = init_GCN_params()
    init_params_GCN, hyper = init_GCN_params_coarsen(L)

    # idx = batch_indices(1)

    # nn_predict_GCN(init_params_GCN, train_images[idx])

    def print_perf_GCN(params, iter, gradient):
        if iter % num_batches == 0:
            train_acc = accuracy_GCN(params, train_images, train_labels)
            test_acc  = accuracy_GCN(params, test_images, test_labels)
            print("{:15}|{:20}|{:20}".format(iter//num_batches, train_acc, test_acc))

            print('{:1.3e}'.format(np.linalg.norm(params['W1'])))
            print('{:1.3e}'.format(np.linalg.norm(params['b1'])))
            print('{:1.3e}'.format(np.linalg.norm(params['W2'])))
            print('{:1.3e}'.format(np.linalg.norm(params['b2'])))



    def objective_GCN(params, iter):
        idx = batch_indices(iter)
        return -log_posterior_GCN(params, train_images[idx], train_labels[idx], L2_reg)

    # Get gradient of objective using autograd.
    objective_GCN_grad = grad(objective_GCN)

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam(objective_GCN_grad, init_params_GCN, step_size=step_size,
                            num_iters=num_epochs * num_batches, callback=print_perf_GCN)
