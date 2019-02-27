from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from tgcn.nn.gcn import TGCNCheb, TGCNCheb_H, GCNCheb, gcn_pool
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import autograd.numpy as npa
# from data import load_mnist
from load.data_hcp import load_hcp_example
import gcn.graph as graph
import gcn.coarsening as coarsening
import sklearn.metrics
from scipy.sparse import coo_matrix
import time, math, random
from torch_geometric.utils import degree, remove_self_loops
from torch_scatter import scatter_add
from torch.nn import Parameter
import scipy.sparse as sp


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


def gcn_pool_4(x):
    x = torch.reshape(x, [x.shape[0], int(x.shape[1] / 4), 4, x.shape[2]])
    x = torch.max(x, dim=2)[0]
    return x


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


def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


class NetTGCN(torch.nn.Module):
    def __init__(self, graphs, coos):
        super(NetTGCN, self).__init__()

        f1, g1, k1, h1 = 1, 96, 10, 15
        #f1, g1, k1 = 1, 32, 25
        self.conv1 = ChebTimeConv(f1, g1, K=k1, H=h1)

        self.drop1 = nn.Dropout(0.1)

        g2, k2 = 96, 10
        self.conv2 = ChebConv(g1, g2, K=k2)

        n2 = graphs[2].shape[0]
        #self.fc1 = torch.nn.Linear(n1 * g1, 10)

        d = 512
        self.fc1 = torch.nn.Linear(int(n2 * g2), d)

        self.dense1_bn = nn.BatchNorm1d(d)
        self.drop2 = nn.Dropout(0.5)

        c = 6
        self.fc2 = torch.nn.Linear(d, c)

        self.coos = coos

    def forward(self, x):
        x = torch.tensor(npa.real(npa.fft.fft(x.to('cpu').numpy(), axis=2))).to('cuda')
        x, edge_index = x, self.coos[0].to(x.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = gcn_pool_4(x)

        x = self.drop1(x)

        edge_index = self.coos[2].to(x.device)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = gcn_pool_4(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        x = self.dense1_bn(x)
        x = F.relu(x)
        x = self.drop2(x)

        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


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


def create_graph(device, shuffled=False):
    def grid_graph(m, corners=False):
        z = graph.grid(m)
        dist, idx = graph.distance_sklearn_metrics(z, k=number_edges, metric=metric)
        A = graph.adjacency(dist, idx)

        if shuffled:
            B = A.toarray()
            B = list(B[np.triu_indices(A.shape[0])])
            random.shuffle(B)
            A = np.zeros((A.shape[0], A.shape[0]))
            indices = np.triu_indices(A.shape[0])
            A[indices] = B
            A = A + A.T - np.diag(A.diagonal())
            A = sp.csr_matrix(A)

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
    #A = graph.replace_random_edges(A, 0)
    graphs, perm = coarsening.coarsen(A, levels=coarsening_levels, self_connections=False)
    return graphs, perm


def load_hcp_tcgn(device):

    time_series, labels, As = load_hcp_example()

    normalized_laplacian = True
    coarsening_levels = 4

    graphs, perm = coarsening.coarsen(As[0], levels=coarsening_levels, self_connections=False)
    coos = [torch.tensor([graph.tocoo().row, graph.tocoo().col], dtype=torch.long).to(device) for graph in graphs]

    idx_train = range(int(0.8*time_series.shape[0]))
    print('Size of train set: {}'.format(len(idx_train)))

    idx_test = range(len(idx_train), time_series.shape[0])
    print('Size of test set: {}'.format(len(idx_test)))

    train_data = time_series[idx_train]
    train_labels = labels[idx_train]
    test_data = time_series[idx_test]
    test_labels = labels[idx_test]

    train_data = perm_data_time(train_data, perm)
    test_data = perm_data_time(test_data, perm)

    return graphs, coos, train_data, test_data, train_labels, test_labels


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = torch.argmax(target, dim=1)
        k = 1.
        w = torch.tensor([1., k, k, k, k, k]).to(device)
        loss = F.nll_loss(output, target, weight=w)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, t1):
    model.eval()
    test_loss = 0
    correct = 0
    cm = 0
    preds = torch.empty(0, dtype=torch.long).to(device)
    targets = torch.empty(0, dtype=torch.long).to(device)
    with torch.no_grad():
        for data_t, target_t in test_loader:
            data = data_t.to(device)
            target = target_t.to(device)
            output = model(data)
            target = torch.argmax(target, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            preds = torch.cat((pred, preds))
            targets = torch.cat((target, targets))
            # cm = sklearn.metrics.confusion_matrix(target, pred)
            cm += sklearn.metrics.confusion_matrix(target.to('cpu').numpy(), pred.to('cpu').numpy())
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    t2 = time.time()
    print(t2-t1)
    print(cm)
    print(cm.sum())
    print(sklearn.metrics.classification_report(targets.to('cpu').numpy(), preds.to('cpu').numpy()))


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, images, labels):
        'Initialization'
        self.labels = labels
        self.images = images

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # X = torch.tensor(self.images[index], dtype=torch.float)
        X = self.images[index].astype('float32')
        # Load data and get label
        y = self.labels[index].astype('float32')

        return X, y


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    #device = torch.device("cpu")

    time_series, labels, As = load_hcp_example()

    normalized_laplacian = True
    coarsening_levels = 4



    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    graphs, coos, train_images, test_images, train_labels, test_labels = load_hcp_tcgn(device)

    training_set = Dataset(train_images, train_labels)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=False)

    validation_set = Dataset(test_images, test_labels)
    test_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)

    model = NetTGCN(graphs, coos)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        t1 = time.time()
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, t1)

    if args.save_model:
        torch.save(model.state_dict(), "hcp_cnn_1gpu2.pt")


if __name__ == '__main__':
    main()



