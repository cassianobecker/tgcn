import torch
from torch.nn import Parameter
# from torch_sparse import spmm
# from torch_geometric.utils import degree, remove_self_loops

# from ..inits import uniform

import math

class GCNCheb(torch.nn.Module):

    def __init__(self, L, in_channels, out_channels, K, bias=True):
        super(GCNCheb, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        # ADD LAPLACIAN AS A MEMBER VARIABLE
        self.L = L

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
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

        # NEED TO IMPLEMENT FORWARD OPERATION WITH CHEBYSHEV VIA SPARSE MULTIPLICATION (or dense in a first moment)
        # MAYBE A CALL TO _chebyshev below

        Tx_0 = x
        out = torch.mm(Tx_0, self.weight[0])

        if K > 1:
            Tx_1 = spmm(edge_index, lap, num_nodes, x)
            out = out + torch.mm(Tx_1, self.weight[1])

        for k in range(2, K):
            Tx_2 = 2 * spmm(edge_index, lap, num_nodes, Tx_1) - Tx_0
            out = out + torch.mm(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))


    # TRANSCRIBED METHOD FROM previous GCN operation (needs to be translated to PYTORCH)
    def _chebyshev(X):
        """Return T_k X where T_k are the Chebyshev polynomials of order up to K.
        Complexity is O(KMN)."""
        dims = list(X.shape)
        dims = tuple([K] + dims)

        Xt = np.empty(dims, L.dtype)

        try:
            X = X._value
        except:
            pass

        Xt[0, ...] = X

        if len(dims) == 3:
            # Xt_1 = T_1 X = L X.
            if K > 1:
                X = self.L.dot(X)
                Xt[1, ...] = X
            # Xt_k = 2 L Xt_k-1 - Xt_k-2.
            for k in range(2, K):
                X = self.L.dot(X)
                Xt[k, ...] = 2 * X - Xt[k - 2, ...]
            return Xt


def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)