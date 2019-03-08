from __future__ import print_function
import sys
sys.path.insert(0, '..')

import torch
from pytorch_based.pytorch_mnist_tgcn import NetTGCN
import autograd.numpy as npa
import torch.nn.functional as F


class NetTGCN_MNIST_Saliency(NetTGCN):

    def class_probabilities(self, x):
        x = torch.tensor(npa.real(npa.fft.fft(x.to('cpu').detach().numpy(), axis=2))).to('cuda')
        x = self.tgcn1(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x

def load_mnist_tgcn_saliency_model(model_fp, graph_fp, device='cuda'):
    L = torch.load(graph_fp)
    model = NetTGCN_MNIST_Saliency(L)
    model.load_state_dict(torch.load(model_fp))
    model.eval()
    model.to(device)
    return model