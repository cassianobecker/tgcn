import os.path as osp

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.data import Data, DataLoader
#from torch.utils.data import DataLoader
import gcn.graph as graph
import gcn.coarsening as coarsening

import torch.optim as optim
import time, random, os
import numpy as np
from load.data import load_mnist

import os


class Net(torch.nn.Module):
    def __init__(self, sh):
        super(Net, self).__init__()
        #self.conv1 = GCNConv(dataset.num_features, 16, improved=False)
        #self.conv2 = GCNConv(16, dataset.num_classes, improved=False)
        self.conv1 = ChebConv(sh[1], 50, K=5)
        #self.conv2 = ChebConv(16, sh[1], K=5)
        self.fc1 = torch.nn.Linear(50, 10)

    def forward(self, data, graphs):

        coos = [torch.tensor([graph.tocoo().row, graph.tocoo().col], dtype=torch.long) for graph in graphs]

        x, edge_index = data.x, coos[0]
        try:
            x = F.relu(self.conv1(x, edge_index))
        except:
            print("boo")
        #x = F.dropout(x, training=self.training)
        #x = self.conv2(x, edge_index)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, graphs, optimizer):
    model.train()
    for data_t in train_loader:

        data = data_t.to(device)

        optimizer.zero_grad()
        output = model(data, graphs)
        #target = torch.argmax(target, dim=1)
        loss = F.nll_loss(output, data.y)
        for p in model.named_parameters():
            if p[0].split('.')[0][:2] == 'fc':
                loss = loss + args.reg_weight*(p[1]**2).sum()
        loss.backward()
        optimizer.step()


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


def get_mnist_data_gcn(perm):

    N, train_data, train_labels, test_data, test_labels = load_mnist(onehot=False)

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
    #L = [torch.tensor(graph.rescale_L(graph.laplacian(A, normalized=normalized_laplacian).todense(), lmax=2), dtype=torch.float).to(device) for A in graphs]
    # graph.plot_spectrum(L)
    #del A

    return graphs, perm

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, images, labels, graphs):
        'Initialization'



        data_list = []

        for i in range(len(images)):
            data = Data(x=torch.tensor([images[i]]).float(), y=torch.tensor([labels[i]], dtype=torch.long))
            data_list.append(data)

        self.data_list = data_list
        self.transform = None

  def __len__(self):
        'Denotes the total number of samples'
        #return len(self.images)
        return len(self.data_list)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        #X = self.images[index].astype('float32')
        #y = self.labels[index].astype('float32')
        data = self.data_list[index]
        data = data if self.transform is None else self.transform(data)
        return data

        #return X, y


def experiment(args):

    args.reg_weight = 5.e-4 #1.e-5

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    graphs, perm = create_graph(device)

    train_images, test_images, train_labels, test_labels = get_mnist_data_gcn(perm)

    train_dataset = Dataset(train_images, train_labels, graphs)
    test_dataset = Dataset(test_images, test_labels, graphs)
    train_loader = DataLoader(train_dataset, batch_size=976, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=976)

    sh = train_images.shape

    model = Net(sh)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)


    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(1, args.epochs):
        train(args, model, device, train_loader, graphs, optimizer)
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