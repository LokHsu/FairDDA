import torch
import torch.nn as nn

from params import args

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, norm_adj):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.norm_adj = norm_adj
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers

        self._init_embeddings()


    def _init_embeddings(self,):
        self.mlp = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.Tanh(),
            nn.Linear(self.emb_size, self.emb_size),
            nn.Sigmoid()
        )
        self.embeddings = nn.ModuleDict()
        self.embeddings['user_embeddings'] = nn.Embedding(self.n_users, self.emb_size)
        self.embeddings['item_embeddings'] = nn.Embedding(self.n_items, self.emb_size)
        nn.init.xavier_uniform_(self.embeddings['user_embeddings'].weight)
        nn.init.xavier_uniform_(self.embeddings['item_embeddings'].weight)


    def forward(self,):
        return self.propagate(
            self.norm_adj,
            self.embeddings['user_embeddings'].weight,
            self.embeddings['item_embeddings'].weight
        )


    def propagate(self, adj, user_emb, item_emb):
        ego_embeddings = torch.cat([user_emb, item_emb], dim=0)
        all_embeddings = [ego_embeddings]

        for _ in range(0, self.n_layers):
            if adj.is_sparse is True:
                ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            else:
                ego_embeddings = torch.mm(adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        return torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)


    def feature_aug(self, adj, e_du, e_di, e_su, e_si):
        mask_u_emb, mask_i_emb = self.feature_mask(
            self.mlp((e_du * e_su)),
            self.mlp((e_di * e_si)),
            self.embeddings['user_embeddings'].weight,
            self.embeddings['item_embeddings'].weight
        )

        ego_embeddings = torch.cat([mask_u_emb, mask_i_emb], dim=0)
        all_embeddings = [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        e_au, e_ai = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        return self.propagate(adj, e_au, e_ai)


    def feature_mask(self, f_u, f_i, e_u0, e_i0):
        mask_u = 1 + (-f_u).exp()
        mask_i = 1 + (-f_i).exp()
        return e_u0 * mask_u, e_i0 * mask_i
