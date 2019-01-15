import torch
from torch.nn import Parameter
# from torch_sparse import spmm
# from torch_geometric.utils import degree, remove_self_loops

# from ..inits import uniform

import math


class TGCNCheb(torch.nn.Module):

    def __init__(self, L, in_channels, out_channels, filter_order, bias=True):
        super(TGCNCheb, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(filter_order, in_channels, out_channels)) # tensor of dimensions k x f x g

        # ADD LAPLACIAN AS A MEMBER VARIABLE
        self.L = L
        self.filter_order = filter_order

        if bias:
            self.bias = Parameter(torch.Tensor(1, L[0].shape[0], out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)


    def forward(self, x):
        """"""
        # Perform filter operation recurrently.

        xc = self._time_chebyshev(x)
        out = torch.einsum("kqnf,kfg->qng", xc, self.weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, filter_order={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))


    def _time_chebyshev(self, X):
        """Return T_k X where T_k are the Chebyshev polynomials of order up to filter_order.
        Complexity is O(KMN).
        self.L: m x n laplacian
        X: q (# examples) x n (vertex count of graph) x f (number of input filters)
        Xt: tensor of dims k (order of chebyshev polynomials) x q x n x f
        """

        #if len(list(X.shape)) == 2:
        #    X = X.unsqueeze(2)

        dims = list(X.shape)
        dims = tuple([self.filter_order] + dims)

        Xt = torch.empty(dims)

        Xt[0, ...] = X

        # Xt_1 = T_1 X = L X.
        if self.filter_order > 1:
            X = torch.einsum("nm,qmf->qnf", self.L, X.float())
            Xt[1, ...] = X
        # Xt_k = 2 L Xt_k-1 - Xt_k-2.
        for k in range(2, self.filter_order):
            #X = Xt[k - 1, ...]
            X = torch.einsum("nm,qmf->qnf", self.L, X.float())
            Xt[k, ...] = 2 * X - Xt[k - 2, ...]
        return Xt


class TGCNCheb_H(torch.nn.Module):

    def __init__(self, L, in_channels, out_channels, filter_order, horizon, bias=True):
        super(TGCNCheb_H, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(filter_order, horizon, in_channels, out_channels)) # tensor of dimensions k x f x g

        # ADD LAPLACIAN AS A MEMBER VARIABLE
        self.L = L
        self.filter_order = filter_order

        if bias:
            self.bias = Parameter(torch.Tensor(1, L[0].shape[0], out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)


    def forward(self, x):
        """"""
        # Perform filter operation recurrently.

        xc = self._time_chebyshev(x)
        out = torch.einsum("kqnhf,khfg->qng", xc, self.weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, filter_order={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))


    def _time_chebyshev(self, X):
        """Return T_k X where T_k are the Chebyshev polynomials of order up to filter_order.
        Complexity is O(KMN).
        self.L: m x n Laplacian
        X: q (# examples) x n (vertex count of graph) x f (number of input filters)
        Xt: tensor of dims k (order of chebyshev polynomials) x q x n x f
        """

        if len(list(X.shape)) == 3:
            X = X.unsqueeze(3)

        dims = list(X.shape)
        dims = tuple([self.filter_order] + dims)

        Xt = torch.empty(dims).to(X.device)

        Xt[0, ...] = X

        # Xt_1 = T_1 X = L X.
        if self.filter_order > 1:
            X = torch.einsum("nm,qmhf->qnhf", self.L, X)
            Xt[1, ...] = X
        # Xt_k = 2 L Xt_k-1 - Xt_k-2.
        for k in range(2, self.filter_order):
            #X = Xt[k - 1, ...]
            X = torch.einsum("nm,qmhf->qnhf", self.L, X)
            Xt[k, ...] = 2 * X - Xt[k - 2, ...]
        return Xt



class GCNCheb(torch.nn.Module):

    def __init__(self, L, in_channels, out_channels, filter_order, bias=True):
        super(GCNCheb, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(filter_order, in_channels, out_channels)) # tensor of dimensions k x f x g

        # ADD LAPLACIAN AS A MEMBER VARIABLE
        self.L = L
        self.filter_order = filter_order

        if bias:
            self.bias = Parameter(torch.Tensor(1, L[0].shape[0], out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)


    def forward(self, x):
        """"""
        # Perform filter operation recurrently.

        xc = self._chebyshev(x)
        out = torch.einsum("kqnf,kfg->qng", xc, self.weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, filter_order={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))


    def _chebyshev(self, X):
        """Return T_k X where T_k are the Chebyshev polynomials of order up to filter_order.
        Complexity is O(KMN).
        self.L: m x n laplacian
        X: q (# examples) x n (vertex count of graph) x f (number of input filters)
        Xt: tensor of dims k (order of chebyshev polynomials) x q x n x f
        """

        if len(list(X.shape)) == 2:
            X = X.unsqueeze(2)

        dims = list(X.shape)
        dims = tuple([self.filter_order] + dims)

        Xt = torch.empty(dims, dtype=torch.float).to(X.device)

        Xt[0, ...] = X

        # Xt_1 = T_1 X = L X.
        # L = torch.Tensor(self.L)
        if self.filter_order > 1:
            X = torch.einsum("nm,qmf->qnf", self.L, X)
            Xt[1, ...] = X
        # Xt_k = 2 L Xt_k-1 - Xt_k-2.
        for k in range(2, self.filter_order):
            #X = Xt[k - 1, ...]
            X = torch.einsum("nm,qmf->qnf", self.L, X)
            Xt[k, ...] = 2 * X - Xt[k - 2, ...]
        return Xt


def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def gcn_pool(x):
    x = torch.reshape(x, [x.shape[0], int(x.shape[1] / 2), 2, x.shape[2]])
    x = torch.max(x, dim=2)[0]
    return x
