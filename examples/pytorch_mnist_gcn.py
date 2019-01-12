from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from tgcn.nn.gcn import GCNCheb, gcn_pool
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from data import load_mnist

import gcn.graph as graph
import gcn.coarsening as coarsening


def get_mnist_data_gcn(perm):

    N, train_data, train_labels, test_data, test_labels = load_mnist()

    idx_train = range(1, 2*512)
    idx_test = range(1, 2*512)

    train_data = train_data[idx_train]
    train_labels = train_labels[idx_train]
    test_data = test_data[idx_test]
    test_labels = test_labels[idx_test]

    train_data = coarsening.perm_data(train_data, perm)
    test_data = coarsening.perm_data(test_data, perm)

    del perm

    return train_data, test_data, train_labels, test_labels


class NetGCN(nn.Module):

    def __init__(self, L):
        super(NetGCN, self).__init__()
        # f: number of input filters
        # g: number of output layers
        # k: order of chebyshev polynomials
        # c: number of classes
        # n: number of vertices at coarsening level

        f1, g1, k1 = 1, 15, 12
        self.gcn1 = GCNCheb(L[0], f1, g1, k1)

        n2 = L[1].shape[0]
        f2, g2, k2 = 15, 25, 12
        self.gcn2 = GCNCheb(L[1], f2, g2, k2)

        c = 10
        self.fc1 = nn.Linear(n2 * g2, c)

    def forward(self, x):
        x = self.gcn1(x)
        x = F.relu(x)
        x = gcn_pool(x)
        x = self.gcn2(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        return F.log_softmax(x, dim=1)


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

    number_edges= 12
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


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = torch.argmax(target, dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            target = torch.argmax(target, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = torch.tensor(self.list_IDs[index])
        # Load data and get label
        y = self.labels[index]

        return X, y


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
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

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    L, perm = create_graph()

    train_images, test_images, train_labels, test_labels = get_mnist_data_gcn(perm)

    training_set = Dataset(train_images, train_labels)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, **kwargs)

    validation_set = Dataset(test_images, test_labels)
    test_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, **kwargs)

    L_tensor = list()
    for m in L:
        coo = m.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        m_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
        L_tensor.append(m_tensor)
    model = NetGCN(L_tensor).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
