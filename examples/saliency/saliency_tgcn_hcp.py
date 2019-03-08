from __future__ import print_function
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import torch
from saliency import gradient_ascent, unpermute
from pytorch_hcp_tgcn import NetTGCN


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

def load_hcp_tgcn_saliency_model(model_state, L, device='cuda'):
    model = NetTGCN_Saliency(L)
    model.load_state_dict(model_state)
    model.eval()
    model.to(device)
    return model

def load_saved_state():
    model_fp = "saved_models/hcp_tgcn.pt"
    graph_fp = "saved_models/hcp_tgcn_laplacian.torch"
    perm_fp = "saved_models/hcp_tgcn_laplacian.torch"
    model_state = torch.load(model_fp)
    L = torch.load(graph_fp)
    perm = torch.load(perm_fp)
    return model_state, L, perm


def main():
    model_state, L, perm = load_saved_state()
    model = load_hcp_tgcn_saliency_model(model_state, L)
    return model



if __name__ == "__main__":
    main()