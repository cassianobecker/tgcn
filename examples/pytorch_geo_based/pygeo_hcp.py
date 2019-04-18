from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
from tgcn.nn.gcn import gcn_pool, gcn_pool_4, ChebConv, ChebTimeConv
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import autograd.numpy as npa
from load.data_hcp import load_hcp_example, StreamDataset, TestDataset, FullDataset
import gcn.graph as graph
import gcn.coarsening as coarsening
import sklearn.metrics
import time, math, random, os
import scipy.sparse as sp
from sys import getsizeof


class NetMLP(nn.Module):

    def __init__(self, sh):

        super(NetMLP, self).__init__()

        c = 512
        self.fc1 = torch.nn.Linear(sh, c)

        d = 1024
        self.fc2 = torch.nn.Linear(c, d)

        e= 750
        self.fc3 = nn.Linear(d, e)

        f=15000
        self.fc4 = nn.Linear(e, f)

        g = 400
        self.fc5 = nn.Linear(f, g)

        h = 6
        self.fc6 = nn.Linear(g, h)


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        #x = F.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return F.log_softmax(x, dim=1)


class NetTGCN(torch.nn.Module):
    def __init__(self, graphs, coos):
        super(NetTGCN, self).__init__()

        f1, g1, k1, h1 = 1, 32, 25, 15
        self.conv1 = ChebTimeConv(f1, g1, K=k1, H=h1)

        self.drop1 = nn.Dropout(0.1)

        g2, k2 = 64, 25
        self.conv2 = ChebConv(g1, g2, K=k2)

        n2 = graphs[0].shape[0]

        c = 6#512
        self.fc1 = torch.nn.Linear(int(n2 * g2), c)

        #self.dense1_bn = nn.BatchNorm1d(d)
        #self.drop2 = nn.Dropout(0.5)

        d = 6
        self.fc2 = torch.nn.Linear(c, d)

        self.coos = coos

    def forward(self, x):
        #x = torch.tensor(npa.real(npa.fft.fft(x.to('cpu').numpy(), axis=2))).to('cuda')
        x, edge_index = x, self.coos[0].to(x.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = gcn_pool_4(x)

        #x = self.drop1(x)

        edge_index = self.coos[0].to(x.device)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = gcn_pool_4(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        #x = self.dense1_bn(x)
        #x = F.relu(x)
        #x = self.drop2(x)

        #x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class NetTGCNBasic(torch.nn.Module):
    def __init__(self, mat_size):
        super(NetTGCNBasic, self).__init__()

        f1, g1, k1, h1 = 1, 64, 25, 15
        self.conv1 = ChebTimeConv(f1, g1, K=k1, H=h1)

        n2 = mat_size

        c = 6
        self.fc1 = torch.nn.Linear(int(n2 * g1), c)

        self.coos = None
        self.perm = None

    def add_graph(self, coos, perm):
        self.coos = coos
        self.perm = perm

    def forward(self, x):
        #x = torch.tensor(npa.real(npa.fft.fft(x.to('cpu').numpy(), axis=2))).to('cuda')
        x, edge_index = x, self.coos[0]#.to(x.device)
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

    time_series, labels, As = load_hcp_example(full=True)

    normalized_laplacian = True
    coarsening_levels = 4
    shuffled=False

    A = As[0]
    #A = arr.todense()

    if shuffled:
        B = A.toarray()
        B = list(B[np.triu_indices(A.shape[0])])
        random.shuffle(B)
        A = np.zeros((A.shape[0], A.shape[0]))
        indices = np.triu_indices(A.shape[0])
        A[indices] = B
        A = A + A.T - np.diag(A.diagonal())
        A = sp.csr_matrix(A)
        graphs, perm = coarsening.coarsen(A, levels=coarsening_levels, self_connections=False)
    #else:
    W = sp.random(As[0].shape[0], As[0].shape[0], density=0, format='csr',
                      data_rvs=lambda s: np.random.uniform(0, 1, size=s))

    #graphs, perm = coarsening.coarsen(W, levels=coarsening_levels, self_connections=False)
    #graphs = [As[0]]
    graphs = [W]

    coos = [torch.tensor([graph.tocoo().row, graph.tocoo().col], dtype=torch.long).to(device) for graph in graphs]

    idx_train = range(int(0.8*time_series.shape[0]))
    print('Size of train set: {}'.format(len(idx_train)))

    idx_test = range(len(idx_train), time_series.shape[0])
    print('Size of test set: {}'.format(len(idx_test)))

    train_data = time_series[idx_train]
    train_labels = labels[idx_train]
    test_data = time_series[idx_test]
    test_labels = labels[idx_test]

    #train_data = perm_data_time(train_data, perm)
    #test_data = perm_data_time(test_data, perm)

    return graphs, coos, train_data, test_data, train_labels, test_labels


def train(args, model, device, train_loader, optimizer, epoch, verbose=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = torch.argmax(target, dim=1)
        k = 1.
        w = torch.tensor([1., k, k, k, k, k]).to(device)
        loss = F.nll_loss(output, target, weight=w)
        for p in model.named_parameters():
            if p[0].split('.')[0][:2] == 'fc':
                loss = loss + args.reg_weight*(p[1]**2).sum()
        loss.backward()
        optimizer.step()
        if verbose:
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))


def train_minibatch(args, model, device, train_loader, optimizer, epoch, verbose=False):
    train_loss = 0
    model.train()
    minibatch = 10
    for batch_idx, (data, target, coos, perm) in enumerate(train_loader):
        #data, target = data.to(device), target.to(device)
        #F = 1024 ** 2
        #print('Bytes of Data: {:1.4f} MB.'.format(getsizeof(data) / F))
        coos = [c[0].to(device) for c in coos]
        #ctr = 0
        target = target.to(device)
        temp_loss = 0
        model.module.add_graph(coos, perm)

        for i in range(len(data)):
            optimizer.zero_grad()
            output = model(data[i].to(device))
            expected = torch.argmax(target[:, i], dim=1)
            k = 1.
            w = torch.tensor([1., k, k, k, k, k]).to(device)
            loss = F.nll_loss(output, expected, weight=w)

            loss = loss / minibatch

            train_loss += loss
            temp_loss += loss
            for p in model.named_parameters():
                if p[0].split('.')[0][:2] == 'fc':
                    loss = loss + args.reg_weight*(p[1]**2).sum()

            loss.backward()

            if batch_idx % minibatch == 0:
                optimizer.step()

            #ctr += 1

        if verbose:
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), temp_loss.item()))

    train_loss /= len(train_loader.dataset)
    return train_loss


def test(args, model, device, test_loader, t1, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    preds = torch.empty(0, dtype=torch.long).to(device)
    targets = torch.empty(0, dtype=torch.long).to(device)
    with torch.no_grad():
        for data_t, target_t, coos, perm in test_loader:
            coos = [c[0].to(device) for c in coos]
            #data = data_t[0].to(device)
            target = target_t.to(device)

            model.module.add_graph(coos, perm)
            for i in range(len(target)):
                output = model(data_t[i].to(device))
                expected = torch.argmax(target[:, i], dim=1)
                test_loss += F.nll_loss(output, expected, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                preds = torch.cat((pred, preds))
                targets = torch.cat((expected, targets))
                correct += pred.eq(expected.view_as(pred)).sum().item()

            #output = model(data)
            #target = torch.argmax(target, dim=1)
            #test_loss += F.nll_loss(output, expected, reduction='sum').item()  # sum up batch loss


    test_loss /= (len(test_loader.dataset) * 270)

    # print('Test Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
    #     epoch, test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))

    return test_loss, correct
    #print(sklearn.metrics.classification_report(targets.to('cpu').numpy(), preds.to('cpu').numpy()))


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


def decode(y_hat, H, Gp, Gn):
    T = len(y_hat)
    N = T + H - 1
    y_decoded = [0] * N
    i = 0

    while (i < T - 3):
        num_agree = ((int(y_hat[i] == y_hat[i + 1] != 0)) +
                     (int(y_hat[i] == y_hat[i + 2] != 0)) +
                     (int(y_hat[i] == y_hat[i + 3] != 0)))
        if (num_agree > 1):
            y_decoded[i + Gn] = y_hat[i]
            i += H
        else:
            i += 1
    return np.array(y_decoded)


def holdout_delabeled(model, data):
    H = 12
    y_holdout = model(data)
    y_holdout = np.reshape(y_holdout, (20, 284 - H + 1))

    num_patients = y_holdout.shape[0]
    y_decoded = np.zeros((num_patients, 284))
    Gp, Gn = 4, 4
    for i in range(y_holdout.shape[0]):
        y_decoded[i, :] = decode(y_holdout[i, :], H, Gp, Gn)

    return y_decoded


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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

    args.reg_weight = 5.e-4
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    #device = torch.device("cpu")

    normalized_laplacian = True
    coarsening_levels = 4



    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    data_type = 'dense'
    if data_type == 'dense':
        mat_size = 77712

    train_set = FullDataset(device, data_type, test=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    #for batch_idx, (data, target, coos, perm) in enumerate(train_loader):
    #    data, target = data.to(device), target.to(device)


    # train_set = StreamDataset()
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    # graphs, coos, perm = train_set.get_graphs(device)

    test_set = FullDataset(device, test=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # graphs, coos, train_images, test_images, train_labels, test_labels = load_hcp_tcgn(device)
    #
    # training_set = Dataset(train_images, train_labels)
    # train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=False)
    #
    # validation_set = Dataset(test_images, test_labels)
    # test_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)


    model = NetTGCNBasic(mat_size)
    #model = NetMLP(int(graphs[0].shape[0] * 15))

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    for epoch in range(1, args.epochs + 1):
        t1 = time.time()
        train_loss = train_minibatch(args, model, device, train_loader, optimizer, epoch, verbose=True)
        scheduler.step()
        test_loss, correct = test(args, model, device, test_loader, t1, epoch)

        print('Epoch: {} Training loss: {:1.3e}, Test loss: {:1.3e}, Accuracy: {}/{} ({:.2f}%)'.format(
            epoch, train_loss, test_loss, correct, len(test_loader.dataset) * 270,
            100. * correct / (len(test_loader.dataset) * 270)))

    if args.save_model:
        torch.save(model.state_dict(), "hcp_cnn_1gpu2.pt")


if __name__ == '__main__':
    seed_everything(76)
    main()



