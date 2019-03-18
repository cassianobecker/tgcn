from __future__ import print_function

import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
#from torch_geometric.nn import GCNConv, ChebConv  # noqa
import argparse
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from load.data import load_mnist
import gcn.graph as graph
import gcn.coarsening as coarsening
import scipy.sparse as sp
import time, random, os
from torch.nn import Parameter
from torch_sparse import spmm
from torch_geometric.utils import degree, remove_self_loops
import math
import autograd.numpy as npa
from tgcn.nn.gcn import gcn_pool, gcn_pool_4, ChebConv, ChebTimeConv


class NetTGCN(torch.nn.Module):
    def __init__(self, graphs, coos):
        super(NetTGCN, self).__init__()

        f1, g1, k1, h1 = 1, 32, 25, 30
        #f1, g1, k1 = 1, 32, 25
        self.conv1 = ChebTimeConv(f1, g1, K=k1, H=h1)

        f2, g2, k2 = 32, 64, 25
        self.conv2 = ChebConv(f2, g2, K=k2)

        n2 = graphs[2].shape[0]
        #self.fc1 = torch.nn.Linear(n1 * g1, 10)

        d = 512
        self.fc1 = torch.nn.Linear(int(n2 * g2), d)

        # self.drop = nn.Dropout(0)

        c = 10
        self.fc2 = torch.nn.Linear(d, c)

        self.coos = coos

    def forward(self, x):
        x = torch.tensor(npa.real(npa.fft.fft(x.to('cpu').numpy(), axis=2))).to('cuda')
        x, edge_index = x, self.coos[0].to(x.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = gcn_pool_4(x)

        edge_index = self.coos[2].to(x.device)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = gcn_pool_4(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class NetTGCNBasic(torch.nn.Module):
    def __init__(self, graphs, coos):
        super(NetTGCNBasic, self).__init__()

        f1, g1, k1, h1 = 1, 32, 25, 12
        self.conv1 = ChebTimeConv(f1, g1, K=k1, H=h1)

        n1 = graphs[0].shape[0]
        self.fc1 = torch.nn.Linear(n1 * g1, 10)

        self.coos = coos

    def forward(self, x):
        x = torch.tensor(npa.real(npa.fft.fft(x.to('cpu').numpy(), axis=2))).to('cuda')
        x, edge_index = x, self.coos[0].to(x.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
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


def get_mnist_data_tgcn(perm):

    N, train_data, train_labels, test_data, test_labels = load_mnist()

    H = 12

    train_data = np.transpose(np.tile(train_data, (H, 1, 1)), axes=[1, 2, 0])
    test_data = np.transpose(np.tile(test_data, (H, 1, 1)), axes=[1, 2, 0])

    #train_data = np.random.rand(train_data.shape[0], len(perm) * len(perm), train_data.shape[2])
    #test_data = np.random.rand(test_data.shape[0], len(perm) * len(perm), test_data.shape[2])

    #idx_train = range(30*512)
    #idx_test = range(10*512)

    #train_data = train_data[idx_train]
    #train_labels = train_labels[idx_train]
    #test_data = test_data[idx_test]
    #test_labels = test_labels[idx_test]

    train_data = perm_data_time(train_data, perm)
    test_data = perm_data_time(test_data, perm)

    del perm

    return train_data, test_data, train_labels, test_labels


def get_fake_data_tgcn(perm):
    M = 9000
    T = 10
    H = 30
    l = len(perm)

    train_data = np.random.rand(M, l, H)
    test_data = np.random.rand(T, l, H)

    train_data = perm_data_time(train_data, perm)
    test_data = perm_data_time(test_data, perm)

    train_labels = np.random.randint(0, 5, M)
    test_labels = np.random.randint(0, 5, T)

    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)

    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)

    del perm

    return train_data, test_data, train_labels, test_labels


def create_graph(device):
    div = 25
    N = int(65e3 / div)
    M = 10
    H = 30
    E = 500000 / div
    d = E / N ** 2  # 0.01

    def grid_graph(m, corners=False):
        z = graph.grid(m)
        dist, idx = graph.distance_sklearn_metrics(z, k=number_edges, metric=metric)
        A = graph.adjacency(dist, idx)
        A = sp.random(A.shape[0], A.shape[0], density=d, format="csr", data_rvs=lambda s: np.random.uniform(0, 0.5, size=s))
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

    #N = 28
    A = grid_graph(int(math.sqrt(N)), corners=False)
    #A = graph.replace_random_edges(A, 0)
    graphs, perm = coarsening.coarsen(A, levels=coarsening_levels, self_connections=False)
    return graphs, perm

def grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def train(args, model, device, train_loader, optimizer, epoch):
    torch.cuda.synchronize()
    #t1 = time.time()
    model.train()
    for batch_idx, (data_t, target_t) in enumerate(train_loader):
        data = data_t.to(device)
        target = target_t.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = torch.argmax(target, dim=1)
        loss = F.nll_loss(output, target)
        for p in model.named_parameters():
            if p[0].split('.')[0][:2] == 'fc':
                loss = loss + args.reg_weight*(p[1]**2).sum()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:

            print('Gradient norm: {:2.4e}'.format(grad_norm(model)))

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    torch.cuda.synchronize()
    #t2 = time.time()
    #print("TIME")
    #print(t2-t1)

def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data_t, target_t in test_loader:
            # data = torch.tensor(data_t, dtype=torch.float).to(device)
            # target = torch.tensor(target_t, dtype=torch.float).to(device)
            # data, target = data.to(device), target.to(device)
            data = data_t.to(device)
            target = target_t.to(device)
            output = model(data)
            target = torch.argmax(target, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(test_loader.dataset)

    print('\nEpoch: {}, AvgLoss: {:.4f}, Accuracy: {:.4f}\n'.format(
        epoch, test_loss, correct / len(test_loader.dataset)))


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


def experiment(args):

    args.reg_weight = 5.e-4 #1.e-5

    # torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    #device = torch.device("cpu")
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    graphs, perm = create_graph(device)
    coos = [torch.tensor([graph.tocoo().row, graph.tocoo().col], dtype=torch.long).to(device) for graph in graphs]

    train_images, test_images, train_labels, test_labels = get_fake_data_tgcn(perm)

    training_set = Dataset(train_images, train_labels)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size)

    validation_set = Dataset(test_images, test_labels)
    test_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size)

    #sh = train_images.shape

    model = NetTGCN(graphs, coos)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)
    #model.cuda()


    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(1, args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()
        test(args, model, device, test_loader, epoch)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
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

    experiment(args)


if __name__ == '__main__':
    seed_everything(1234)
    main()