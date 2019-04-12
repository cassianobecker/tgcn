import math
import torch
from torch.nn import Parameter, init
from torch_geometric.utils import degree, remove_self_loops
from torch_scatter import scatter_add


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
        L = self.L.to(X.device)

        Xt[0, ...] = X

        # Xt_1 = T_1 X = L X.
        if self.filter_order > 1:
            X = torch.einsum("nm,qmhf->qnhf", L, X)
            Xt[1, ...] = X
        # Xt_k = 2 L Xt_k-1 - Xt_k-2.
        for k in range(2, self.filter_order):
            #X = Xt[k - 1, ...]
            X = torch.einsum("nm,qmhf->qnhf", L, X)
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
            self.bias = Parameter(torch.Tensor(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #     #bound = 1 / math.sqrt(fan_in)
        #     #init.uniform_(self.bias, -bound, bound)
        #     uniform(fan_in, self.bias)


    def forward(self, x):
        """"""
        # Perform filter operation recurrently.

        xc = self._chebyshev(x)
        out = torch.einsum("kqnf,kfg->qng", xc, self.weight)
        #b = self.bias.cpu().detach().numpy()

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
        L = self.L.to(X.device)

        Xt[0, ...] = X

        # Xt_1 = T_1 X = L X.
        # L = torch.Tensor(self.L)
        if self.filter_order > 1:
            X = torch.einsum("nm,qmf->qnf", L, X)
            Xt[1, ...] = X
        # Xt_k = 2 L Xt_k-1 - Xt_k-2.
        for k in range(2, self.filter_order):
            #X = Xt[k - 1, ...]
            X = torch.einsum("nm,qmf->qnf", L, X)
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


def gcn_pool_4(x):
    x = torch.reshape(x, [x.shape[0], int(x.shape[1] / 4), 4, x.shape[2]])
    x = torch.max(x, dim=2)[0]
    return x


def spmm(index, value, m, matrix):
    """Matrix product of sparse matrix with dense matrix.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of sparse matrix.
        matrix (:class:`Tensor`): The dense matrix.

    :rtype: :class:`Tensor`
    """

    row, col = index
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix[col]
    out = out.permute(-1, 0) * value
    out = out.permute(-1, 0)
    out = scatter_add(out, row, dim=0, dim_size=m)

    return out


def spmm_batch_2(index, value, m, matrix):
    """Matrix product of sparse matrix with dense matrix.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of sparse matrix.
        matrix (:class:`Tensor`): The dense matrix.

    :rtype: :class:`Tensor`
    """

    row, col = index
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix[:, col]
    try:
        sh = out.shape[2]
    except:
        out = out.unsqueeze(-1)
        sh = 1

    #out = out.permute(1, 2, 0)
    #out = torch.mul(out, value.repeat(-1, sh))
    #out = out.permute(1, 2, 0)
    temp = value.expand(sh, value.shape[0]).permute(1, 0)
    out = torch.einsum("abc,bc->abc", out, temp)
    out = scatter_add(out, row, dim=1, dim_size=m)

    return out


def spmm_batch_3(index, value, m, matrix):
    """Matrix product of sparse matrix with dense matrix.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of sparse matrix.
        matrix (:class:`Tensor`): The dense matrix.

    :rtype: :class:`Tensor`
    """

    row, col = index
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix[:, col]
    try:
        sh = out.shape[3:]
        sh = matrix.shape[2:]
    except:
        out = out.unsqueeze(-1)
        sh = matrix.shape[2:]

    #out = out.permute(1, 2, 0)
    #out = torch.mul(out, value.repeat(-1, sh))
    #out = out.permute(1, 2, 0)
    sh = sh + (value.shape[0],)
    temp = value.expand(sh)
    temp = temp.permute(2, 0, 1)
    out = torch.einsum("abcd,bcd->abcd", out, temp)
    out = scatter_add(out, row, dim=1, dim_size=m)

    return out


class ChebConv(torch.nn.Module):
    r"""The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \sum_{k=0}^{K-1} \mathbf{\hat{X}}_k \cdot
        \mathbf{\Theta}_k

    where :math:`\mathbf{\hat{X}}_k` is computed recursively by

    .. math::
        \mathbf{\hat{X}}_0 &= \mathbf{X}

        \mathbf{\hat{X}}_1 &= \mathbf{\hat{L}} \cdot \mathbf{X}

        \mathbf{\hat{X}}_k &= 2 \cdot \mathbf{\hat{L}} \cdot
        \mathbf{\hat{X}}_{k-1} - \mathbf{\hat{X}}_{k-2}

    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size, *i.e.* number of hops.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, K, bias=True):
        super(ChebConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        row, col = edge_index
        num_nodes, num_edges, K = x.size(1), row.size(0), self.weight.size(0)

        if edge_weight is None:
            edge_weight = x.new_ones((num_edges, ))
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        deg = degree(row, num_nodes, dtype=x.dtype)

        # Compute normalized and rescaled Laplacian.
        deg = deg.pow(-0.5)
        deg[deg == float('inf')] = 0
        lap = -deg[row] * edge_weight * deg[col]

        # Perform filter operation recurrently.
        if len(x.shape) < 3:
            Tx_0 = x.unsqueeze(-1)
        else:
            Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0])

        if K > 1:
            Tx_1 = spmm_batch_2(edge_index, lap, num_nodes, Tx_0)
            #Tx_1 = torch.stack([spmm(edge_index, lap, num_nodes, Tx_0[i]) for i in range(x.shape[0])])
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, K):
            temp = spmm_batch_2(edge_index, lap, num_nodes, Tx_1)
            #temp = torch.stack([spmm(edge_index, lap, num_nodes, Tx_1[i]) for i in range(x.shape[0])])
            Tx_2 = 2 * temp - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))


class ChebTimeConv(torch.nn.Module):
    r"""The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \sum_{k=0}^{K-1} \mathbf{\hat{X}}_k \cdot
        \mathbf{\Theta}_k

    where :math:`\mathbf{\hat{X}}_k` is computed recursively by

    .. math::
        \mathbf{\hat{X}}_0 &= \mathbf{X}

        \mathbf{\hat{X}}_1 &= \mathbf{\hat{L}} \cdot \mathbf{X}

        \mathbf{\hat{X}}_k &= 2 \cdot \mathbf{\hat{L}} \cdot
        \mathbf{\hat{X}}_{k-1} - \mathbf{\hat{X}}_{k-2}

    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size, *i.e.* number of hops.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, K, H, bias=True):
        super(ChebTimeConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(K, H, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        row, col = edge_index
        num_nodes, num_edges, K = x.size(1), row.size(0), self.weight.size(0)

        if edge_weight is None:
            edge_weight = x.new_ones((num_edges, ))
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        deg = degree(row, num_nodes, dtype=x.dtype)

        # Compute normalized and rescaled Laplacian.
        deg = deg.pow(-0.5)
        deg[deg == float('inf')] = 0
        lap = -deg[row] * edge_weight * deg[col]

        # Perform filter operation recurrently.
        if len(x.shape) < 4:
            Tx_0 = x.unsqueeze(-1)
        else:
            Tx_0 = x
        #out = torch.matmul(Tx_0, self.weight[0])

        out = torch.einsum("qnhf,hfg->qng", Tx_0, self.weight[0])
        if K > 1:
            Tx_1 = spmm_batch_3(edge_index, lap, num_nodes, Tx_0)
            out = out + torch.einsum("qnhf,hfg->qng", Tx_1, self.weight[1])

        for k in range(2, K):
            temp = spmm_batch_3(edge_index, lap, num_nodes, Tx_1)
            Tx_2 = 2 * temp - Tx_0
            out = out + torch.einsum("qnhf,hfg->qng", Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))