from __future__ import print_function
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


class NetMLP(nn.Module):

    def __init__(self, sh):

        # f: number of input filters
        # g: number of output filters
        # k: order of chebyshev polynomial
        # c: number of classes
        # n: number of vertices at coarsening level

        super(NetMLP, self).__init__()

        c = 500
        self.fc1 = nn.Linear(sh, c)

        d = 1000
        self.fc2 = nn.Linear(c, d)

        e= 750
        self.fc3 = nn.Linear(d, e)

        f=15000
        self.fc4 = nn.Linear(e, f)

        g = 400
        self.fc5 = nn.Linear(f, g)

        h = 10
        self.fc6 = nn.Linear(g, h)



    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return F.log_softmax(x, dim=1)


class NetGCNBasic(nn.Module):

    def __init__(self, L):

        # f: number of input filters
        # g: number of output filters
        # k: order of chebyshev polynomial
        # c: number of classes
        # n: number of vertices at coarsening level
        super(NetGCNBasic, self).__init__()

        f1, g1, k1 = 1, 10, 25
        self.gcn1 = GCNCheb(L[0], f1, g1, k1)

        n1 = L[0].shape[0]
        d = 10
        self.fc1 = nn.Linear(n1 * g1, d)

    def forward(self, x):

        x = self.gcn1(x)
        x = F.relu(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)


class NetGCN2Layer(nn.Module):

    def __init__(self, L):

        # f: number of input filters
        # g: number of output filters
        # k: order of chebyshev polynomial
        # c: number of classes
        # n: number of vertices at coarsening level
        super(NetGCN2Layer, self).__init__()

        f1, g1, k1 = 1, 32, 25
        self.gcn1 = GCNCheb(L[0], f1, g1, k1)

        f2, g2, k2 = g1, 64, 25
        self.gcn2 = GCNCheb(L[2], f2, g2, k2)

        n2 = L[2].shape[0]
        d = 512
        self.fc1 = nn.Linear(int(n2 * g2 /4), d)

        #self.drop = nn.Dropout(0)

        c = 10
        self.fc2 = nn.Linear(d, c)

    def forward(self, x):

        x = self.gcn1(x)
        x = F.relu(x)
        x = gcn_pool_4(x)

        x = self.gcn2(x)
        x = F.relu(x)
        x = gcn_pool_4(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.drop(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class NetGCN3Layer(nn.Module):

    def __init__(self, L):

        # f: number of input filters
        # g: number of output filters
        # k: order of chebyshev polynomial
        # c: number of classes
        # n: number of vertices at coarsening level
        super(NetGCN3Layer, self).__init__()

        f1, g1, k1 = 1, 32, 25
        self.gcn1 = GCNCheb(L[0], f1, g1, k1)

        self.drop1 = nn.Dropout(0.1)

        f2, g2, k2 = g1, 64, 25
        self.gcn2 = GCNCheb(L[2], f2, g2, k2)
        self.dense1_bn = nn.BatchNorm1d(50)

        f3, g3, k3 = g2, 64, 25
        self.gcn3 = GCNCheb(L[4], f3, g3, k3)
        self.dense1_bn = nn.BatchNorm1d(50)

        n3 = L[2].shape[0]
        d = 512
        self.fc1 = nn.Linear(int(n3 * g3 /4), d)

        self.dense1_bn = nn.BatchNorm1d(d)
        self.drop2 = nn.Dropout(0.5)

        c = 10
        self.fc2 = nn.Linear(d, c)


    def forward(self, x):

        x = self.gcn1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = gcn_pool_4(x)
        x = self.gcn2(x)
        x = F.relu(x)
        x = gcn_pool_4(x)
        x = self.gcn3(x)
        x = F.relu(x)
        #x = gcn_pool_4(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.dense1_bn(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

class NetGCN3(nn.Module):

    def __init__(self, L):

        # f: number of input filters
        # g: number of output filters
        # k: order of chebyshev polynomial
        # c: number of classes
        # n: number of vertices at coarsening level

        super(NetGCN3, self).__init__()

        f1, g1, k1 = 1, 30, 25
        self.gcn1 = GCNCheb(L[0], f1, g1, k1)

        f2, g2, k2 = g1, 20, 25
        self.gcn2 = GCNCheb(L[0], f2, g2, k2)

        f3, g3, k3 = g2, 10, 25
        self.gcn3 = GCNCheb(L[0], f3, g3, k2)

        n3 = L[0].shape[0]
        d = 500
        self.fc1 = nn.Linear(n3 * g3, d)

        c = 10
        self.fc2 = nn.Linear(d, c)



    def forward(self, x):

        x = self.gcn1(x)
        x = F.relu(x)
        # x = gcn_pool(x)
        x = self.gcn2(x)
        x = F.relu(x)
        x = self.gcn3(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


def get_mnist_data_gcn(perm):

    N, train_data, train_labels, test_data, test_labels = load_mnist()

    # sz_batch = 512
    # n_batch_train = 40
    # n_batch_test = 5
    #
    # idx_train = range(0, n_batch_train*sz_batch)
    # idx_test = range(0, n_batch_test*sz_batch)
    #
    # train_data = train_data[idx_train]
    # train_labels = train_labels[idx_train]
    # test_data = test_data[idx_test]
    # test_labels = test_labels[idx_test]

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
    A = graph.replace_random_edges(A, 0)
    graphs, perm = coarsening.coarsen(A, levels=coarsening_levels, self_connections=False)
    L = [torch.tensor(graph.rescale_L(graph.laplacian(A, normalized=normalized_laplacian).todense(), lmax=2), dtype=torch.float).to(device) for A in graphs]
    # graph.plot_spectrum(L)
    del A

    return L, perm

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
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    L, perm = create_graph(device)

    train_images, test_images, train_labels, test_labels = get_mnist_data_gcn(perm)

    training_set = Dataset(train_images, train_labels)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size)

    validation_set = Dataset(test_images, test_labels)
    test_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size)

    # L_tensor = list()
    # for m in L:
    #     coo = m.tocoo()
    #     values = coo.data
    #     indices = np.vstack((coo.row, coo.col))
    #
    #     i = torch.LongTensor(indices)
    #     v = torch.FloatTensor(values)
    #     shape = coo.shape
    #
    #     m_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
    #     L_tensor.append(m_tensor)

    model = NetGCNBasic(L)
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
        torch.save(model.state_dict(), "mnist_cnn.pt")



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