from __future__ import print_function
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import torch
from saliency import gradient_ascent, unpermute
from pytorch_based.pytorch_hcp_tgcn import NetTGCN
import torch.nn.functional as F


class NetTGCN_Saliency(NetTGCN):

    def class_probabilities(self, x):
        x = torch.rfft(x, signal_ndim=1, onesided=False)[:, :, :, 0].to(self.device)
        x = self.tgcn1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = gcn_pool_4(x)
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
    print("model loaded")
    input_shape = (1, 160, 15)
    n_real_nodes = 148
    target_dir = "hcp_saliency_results"
    for target in range(6):
        x_hat = gradient_ascent(model, input_shape=input_shape, target=target)
        x_hat_real = unpermute(x_hat, perm, n_real_nodes)
        output = x_hat_real.reshape(148, 15)
        np.save(output, f"{target_dir}/hcp_saliency_target_{target}")
        print(f"Target: {target} saved to - {target_dir}/hcp_saliency_target_{target}")


if __name__ == "__main__":
    main()