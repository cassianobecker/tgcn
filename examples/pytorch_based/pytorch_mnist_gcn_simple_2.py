from __future__ import print_function
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import argparse
import torch
import torch.nn as nn
from tgcn.nn.gcn import GCNCheb, gcn_pool, gcn_pool_4
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from load.data import load_mnist
import gcn.graph as graph
import gcn.coarsening as coarsening
import scipy.sparse as sp
import time, random, os


def create_graph():
    number_edges= 8
    metric = 'euclidean'
    normalized_laplacian = True
    coarsening_levels = 4

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

    A = grid_graph(28, corners=False)
    L = torch.tensor(graph.rescale_L(graph.laplacian(A, normalized=normalized_laplacian).todense(), lmax=2))

    del A

    return L

class NetGCNBasic(nn.Module):

    L = create_graph()

    def __init__(self):

        # f: number of input filters
        # g: number of output filters
        # k: order of chebyshev polynomial
        # c: number of classes
        # n: number of vertices at coarsening level
        super(NetGCNBasic, self).__init__()

        L = NetGCNBasic.L
        f1, g1, k1 = 1, 10, 8 # 1, 64, 25
        f2, g2, k2 = g1, 20, 4
        self.gcn1 = GCNCheb(L, f1, g1, k1)
        self.gcn2 = GCNCheb(L, f2, g2, k2)

        n1 = L.shape[0]
        d = 10
        self.fc1 = nn.Linear(n1 * g2, d)

    def forward(self, x):

        x = self.gcn1(x)
        x = F.relu(x)

        x = self.gcn2(x)
        x = F.relu(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)

    def class_probabilities(self, x):

        x = self.gcn1(x)
        x = F.relu(x)

        x = self.gcn2(x)
        x = F.relu(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return x



def get_mnist_data_gcn():
    N, train_data, train_labels, test_data, test_labels = load_mnist()
    return train_data, test_data, train_labels, test_labels

def create_graph():
    number_edges= 8
    metric = 'euclidean'
    normalized_laplacian = True
    coarsening_levels = 4

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

    A = grid_graph(28, corners=False)
    L = torch.tensor(graph.rescale_L(graph.laplacian(A, normalized=normalized_laplacian).todense(), lmax=2))

    del A

    return L


def train(args, model, device, train_loader, optimizer, epoch):
    torch.cuda.synchronize()
    #t1 = time.time()
    model.train()
    for batch_idx, (data_t, target_t) in enumerate(train_loader):
        # data, target = data.to(device), target.to(device)
        # data = torch.tensor(data_t, dtype=torch.float).to(device)
        # target = torch.tensor(target_t, dtype=torch.float).to(device)
        data = data_t.to(device)
        target = target_t.to(device)
        # data = torch.from_numpy(data_t).float().to(device)
        # target = torch.from_numpy(target_t).float().to(device)

        optimizer.zero_grad()
        output = model(data)
        target = torch.argmax(target, dim=1)
        loss = F.nll_loss(output, target)
        for p in model.named_parameters():
            if p[0].split('.')[0][:2] == 'fc':
                loss = loss + args.reg_weight*(p[1]**2).sum()
        loss.backward()
        optimizer.step()

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


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

# ###################################
# ########## EXPERIMENT #############
# ###################################


def experiment(args):

    args.reg_weight = 5.e-4 #1.e-5

    # torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    print("cuda" if use_cuda else "cpu")
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_images, test_images, train_labels, test_labels = get_mnist_data_gcn()

    training_set = Dataset(train_images, train_labels)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size)

    validation_set = Dataset(test_images, test_labels)
    test_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size)

    model = NetGCNBasic()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)

    #model.gcn1 = torch.nn.DataParallel(model.gcn1, device_ids=[0, 1])
    #model.gcn2 = torch.nn.DataParallel(model.gcn2, device_ids=[0, 1])
    #model.gcn3 = torch.nn.DataParallel(model.gcn3, device_ids=[0, 1])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(1, args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()
        test(args, model, device, test_loader, epoch)

    if args.save_model:
        model_fp = "mnist_gcn_simple_two_layer.pt"
        torch.save(model.state_dict(), model_fp)
        print(f"Saved model to {model_fp}")



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