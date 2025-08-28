import torch.nn as nn
from .lightgcn import LightGCN

from params import args

class SemiGCN(nn.Module):
    def __init__(self, n_users, n_items, n_sens, norm_adj):
        super(SemiGCN, self).__init__()
        self.gcn = LightGCN(n_users, n_items, norm_adj)
        self.fc = nn.Linear(args.emb_size, n_sens)

    def forward(self, ):
        e_su, e_si = self.gcn()
        s_u = self.fc(e_su)
        s_i = self.fc(e_si)
        return e_su, e_si, s_u, s_i
