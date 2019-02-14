from __future__ import print_function

import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from load.data import load_mnist
import gcn.graph as graph
import gcn.coarsening as coarsening
import scipy.sparse as sp
import time, random, os

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self, graphs, coos):
        super(Net, self).__init__()

        f1, g1, k1 = graphs[0].shape[0], 10, 2
        self.conv1 = ChebConv(f1, g1, K=k1)

        #f2, g2, k2 = 1, 10, 20
        #self.conv2 = ChebConv(16, data.num_features, K=2)

        self.fc1 = torch.nn.Linear(g1, 10)

        self.coos = coos

    def forward(self, x):
        x, edge_index = x, self.coos[0]
        x = F.relu(self.conv1(x, edge_index))
        #x = F.dropout(x, training=self.training)
        #x = self.conv2(x, edge_index)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


def get_mnist_data_gcn(perm):

    N, train_data, train_labels, test_data, test_labels = load_mnist()

    train_data = coarsening.perm_data(train_data, perm)
    test_data = coarsening.perm_data(test_data, perm)

    del perm

    return train_data, test_data, train_labels, test_labels

def create_graph(device):
    def grid_graph(m, corners=False):
        z = graph.grid(m)
        dist, idx = graph.distance_sklearn_metrics(z, k=number_edges, metric=metric)
        A = graph.adjacency(dist, idx)
        #A = sp.random(A.shape[0], A.shape[0], density=0.01, format="csr", data_rvs=lambda s: np.random.uniform(0, 0.5, size=s))
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
        #if batch_idx % args.log_interval == 0:

            #print('Gradient norm: {:2.4e}'.format(grad_norm(model)))
            #
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #            100. * batch_idx / len(train_loader), loss.item()))

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

    #device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    graphs, perm = create_graph(device)
    coos = [torch.tensor([graph.tocoo().row, graph.tocoo().col], dtype=torch.long).to(device) for graph in graphs]

    train_images, test_images, train_labels, test_labels = get_mnist_data_gcn(perm)

    training_set = Dataset(train_images, train_labels)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size)

    validation_set = Dataset(test_images, test_labels)
    test_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size)

    #sh = train_images.shape

    model = Net(graphs, coos)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)


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
    parser.add_argument('--batch-size', type=int, default=976, metavar='N',
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