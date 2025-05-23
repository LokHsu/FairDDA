# -*- coding: utf-8 -*-


import pdb
import torch
import torch.nn.functional as F

def kernel_matrix(x, sigma):
    return torch.exp((torch.matmul(x, x.transpose(0,1)) - 1) / sigma)    ### real_kernel

def hsic(Kx, Ky, m):
    Kxy = torch.mm(Kx, Ky)
    h = torch.trace(Kxy) / m ** 2 + torch.mean(Kx) * torch.mean(Ky) - \
        2 * torch.mean(Kxy) / m
    return h * (m / (m - 1)) ** 2

def calc_ib_loss(u_emb, sen_emb, sigma):
    Kx = kernel_matrix(u_emb, sigma)
    Ky = kernel_matrix(sen_emb, sigma)
    loss = hsic(Kx, Ky, u_emb.shape[0])
    return loss

def bpr_loss(user_emb, pos_emb, neg_emb):
    pos_score = torch.sum(user_emb * pos_emb, dim=1)
    neg_score = torch.sum(user_emb * neg_emb, dim=1)
    mf_loss = torch.mean(F.softplus(neg_score - pos_score))
    emb_loss = (1 / 2) * (user_emb.norm(2).pow(2) +
                          pos_emb.norm(2).pow(2) +
                          neg_emb.norm(2).pow(2)) / user_emb.shape[0]
    return mf_loss, emb_loss


def calc_bpr_loss(user_emb, item_emb, u, i, j):
    batch_user_emb = user_emb[u]
    batch_pos_item_emb = item_emb[i]
    batch_neg_item_emb = item_emb[j]

    mf_loss, emb_loss = bpr_loss(batch_user_emb, batch_pos_item_emb, batch_neg_item_emb)
    return mf_loss, emb_loss


def aug_condition_info_nce_for_embeddings(x, z, p, tau=0.5):
    N = x.shape[0]

    x_norm = F.normalize(x)
    y_norm = F.normalize(z)

    pos_score = torch.sum(x_norm * y_norm, dim=1)
    pos_score = torch.exp(pos_score / tau)

    neg_score = torch.zeros(N, dtype=torch.float32).to(x.device)
    neg_score2 = torch.zeros(N, dtype=torch.float32).to(x.device)
    for cat in set(p.tolist()):
        x_given = x_norm[p == cat]
        y_given = y_norm[p == cat]

        t = x_given @ y_given.T
        t = torch.sum(torch.exp(t / tau), dim=1)
        neg_score[p == cat] = t
        
        
        t2 = y_given @ y_given.T
        t2=t2-torch.diag(t2.diag())
        t2 = torch.sum(torch.exp(t2 / tau), dim=1)
        
        
        neg_score2[p == cat] = t2

    cl_loss = -torch.log(pos_score / (neg_score+neg_score2)).mean()
    return cl_loss


def info_nce_for_embeddings(x, z, tau=0.5):  
    N = x.shape[0]

    x_norm = F.normalize(x)
    y_norm = F.normalize(z)

    pos_score = torch.sum(x_norm * y_norm, dim=1)
    pos_score = torch.exp(pos_score / tau)

    neg_score = torch.zeros(N, dtype=torch.float32).to(x.device)
    neg_score2 = torch.zeros(N, dtype=torch.float32).to(x.device)
    x_given = x_norm
    y_given = y_norm

    t = x_given @ y_given.T
    t = torch.sum(torch.exp(t / tau), dim=1)
    neg_score = t
    
    
    t2 = y_given @ y_given.T
    t2 = torch.sum(torch.exp(t2 / tau), dim=1)
    neg_score2 = t2

    cl_loss = -torch.log(pos_score / (neg_score+neg_score2)).mean()
    return cl_loss