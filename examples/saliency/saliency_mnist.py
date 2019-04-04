from __future__ import print_function
import sys
sys.path.insert(0, '..')

import torch
import numpy as np
# from pytorch_based.pytorch_mnist_tgcn import NetTGCN
from pytorch_based.pytorch_mnist_gcn import NetGCNBasic, NetGCN2Layer
from tgcn.nn.gcn import GCNCheb, gcn_pool, gcn_pool_4
from saliency import *
import autograd.numpy as npa
import torch.nn.functional as F


# class NetTGCN_MNIST_Saliency(NetTGCN):

#     def class_probabilities(self, x):
#         x = torch.tensor(npa.real(npa.fft.fft(x.to('cpu').detach().numpy(), axis=2))).to('cuda')
#         x = self.tgcn1(x)
#         x = F.relu(x)
#         x = x.view(x.shape[0], -1)
#         x = self.fc1(x)
#         return x

class NetGCN_MNIST_Saliency(NetGCNBasic):

    def class_probabilities(self, x):
        x = self.gcn1(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x

class NetGCN_2LAYER_MNIST_Saliency(NetGCN2Layer):

    def class_probabilities(self, x):
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
        return x

def load_mnist_saliency_model(model_fp=None, graph_fp=None, model_state=None, L=None, net_type='tgcn', device='cuda'):
    if L is None:
        L = torch.load(graph_fp)
    if model_state is None:
        model_state = torch.load(model_fp)

    if net_type == 'tgcn':
        model = NetTGCN_MNIST_Saliency(L)
    elif net_type == 'gcn':
        model = NetGCN_MNIST_Saliency(L)
    elif net_type == 'gcn2':
        model = NetGCN_2LAYER_MNIST_Saliency(L)

    model.load_state_dict(model_state)
    model.eval()
    model.to(device)
    return model

def load_saved_state():
    model_fp = '../saved_models/mnist_gcn_tmp/mnist_gcn.pt'
    graph_fp = '../saved_models/mnist_gcn_tmp/mnist_gcn_laplacian.torch'
    perm_fp = '../saved_models/mnist_gcn_tmp/mnist_gcn_perm.torch'
    model_state = torch.load(model_fp)
    L = torch.load(graph_fp)
    perm = torch.load(perm_fp)
    return model_state, L, perm

def main():
    net_type = 'gcn'
    model_state, L, perm = load_saved_state()
    model = load_mnist_saliency_model(model_state=model_state, L=L, net_type=net_type)
    print("model loaded")
    input_shape = (1, 960, 1)
    n_real_nodes = 28 * 28
    fake_nodes = [0, perm[28*28:], 0]
    target_dir = "mnist_saliency_gcn_results2"
    for target in range(10):
        init = torch.rand_like(torch.zeros(input_shape)).to('cuda') # to ensure input is in range 0,1
        init[fake_nodes] = 0
        x_hat = gradient_ascent(model, init=init, target=target, max_iter=1000, ep=.01, reg_penalty=.01,
                                min_val=0, max_val=1, fake_nodes=fake_nodes)
        x_hat_real = unpermute(x_hat, perm, n_real_nodes)
        output = x_hat_real.reshape(28, 28)
        np.save(f"{target_dir}/mnist_gcn_saliency_target_{target}", output)
        print(f"Target: {target} saved to - {target_dir}/hcp_saliency_target_{target}")


if __name__ == "__main__":
    main()