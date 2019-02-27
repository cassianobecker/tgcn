from __future__ import print_function
import sys
sys.path.insert(0, '..')

import torch
from pytorch_hcp_tgcn import NetTGCN
from load.data_hcp import load_hcp_example
import gcn.graph as graph
import gcn.coarsening as coarsening

def load_hcp_laplacian(device='cuda'):
    time_series, labels, As = load_hcp_example()

    normalized_laplacian = True
    coarsening_levels = 4

    graphs, perm = coarsening.coarsen(As[0], levels=coarsening_levels, self_connections=False)
    L = [torch.tensor(graph.rescale_L(graph.laplacian(A, normalized=normalized_laplacian).todense(), lmax=2),
                      dtype=torch.float).to(device) for A in graphs]
    return L


class NetTGCN_Saliency(NetTGCN):

    def class_probabilities(self, x):
        x = torch.rfft(x, signal_ndim=1, onesided=False)[:, :, :, 0].to(self.device)
        x = self.tgcn1(x)
        x = F.relu(x)

        x = self.drop1(x)
        x = self.gcn2(x)
        x = F.relu(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.dense1_bn(x)
        x = F.relu(x)

        x = self.drop2(x)
        x = self.fc2(x)
        return x