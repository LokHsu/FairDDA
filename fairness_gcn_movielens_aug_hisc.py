import argparse
import calendar
import gc
import pickle
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pyro
from tqdm import tqdm
from models import *
from utils import *
from models.my_relaxed_bernoulli import RelaxedBernoulliStraightThrough
import copy
import datetime
import csv

def train_semigcn(gcn, sens, n_users,n_items, e_xu, e_xi, args,classes_num, device='cuda:0'):
    optimizer = optim.Adam(gcn.parameters(), lr=args.lr)
    sens = torch.tensor(sens,dtype=torch.float).to(torch.long)
    sens=sens.to(device)
    start_time = time.time()
    #criterion=nn.L1Loss(reduction='mean')
    criterion=nn.MSELoss(reduction='mean')
    final_loss=0.0
    for epoch in tqdm(range(args.sim_epochs)):
        e_su, e_si, su, si = gcn()
        classify_loss = F.cross_entropy(su.squeeze(), sens.squeeze())
           
        loss=classify_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = classify_loss.item()
    print('%.6f'% (final_loss))
    end_time = time.time() 
    print(end_time-start_time)

def get_final_emb(e_xu, e_xi,e_nu,e_ni):
    user_emb, item_emb = e_xu, e_xi
    mask= 1+(-user_emb).exp()
    mask_i=1+(-item_emb).exp()
    return e_nu*mask,e_ni*mask_i

#feature aug
def propagate( adj,e_xu, e_xi,e_su, e_si,n_users,n_items,e_zu,e_zi,mlp_list):
    all_emb = []
        
    u_emb,i_emb=get_final_emb(mlp_list((e_xu*e_su)), mlp_list((e_xi*e_si)),e_zu[:,0,:],e_zi[:,0,:])
        
    ego_embeddings= torch.cat([u_emb, i_emb], dim=0)
    #ego_embeddings= torch.cat([e_zu[:,0,:],e_zi[:,0,:]], dim=0)
    all_emb += [ego_embeddings]
    
    all_emb = torch.stack(all_emb, dim=1)
    all_emb = torch.mean(all_emb, dim=1)
    aug_user_emb,aug_item_emb  = torch.split(all_emb, [n_users, n_items], dim=0)
    
    ego_embeddings = torch.cat([aug_user_emb, aug_item_emb], dim=0)
    all_embeddings = [ego_embeddings]

    for k in range(1, 4):
        if adj.is_sparse is True:
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
        else:
            ego_embeddings = torch.mm(adj, ego_embeddings)
        all_embeddings += [ego_embeddings]

    all_embeddings = torch.stack(all_embeddings, dim=1)
    all_embeddings = torch.mean(all_embeddings, dim=1)
    u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [n_users, n_items], dim=0)

    return u_g_embeddings, i_g_embeddings


def train_unify_mi(sens_enc, inter_enc, e_xu, e_xi, dataset, u_sens,
                   n_users, n_items, train_u2i, test_u2i, args):
    mlp_list=list()
    params_list=[]
    start_time = time.time()
    mlp_feature=nn.Sequential(
        nn.Linear(64, 64),
        nn.Tanh(),
        #nn.Dropout(p=0.2),
        nn.Linear(64, 64),
        nn.Sigmoid()).to(args.device)
    mlp_graph=nn.Sequential(
        nn.Linear(64, 64),
        nn.Tanh(),
        #nn.Dropout(p=0.2),
        nn.Linear(64, 64),
        nn.Sigmoid()).to(args.device)
    # +list(mlp_graph.parameters())
    optimizer_G = optim.Adam(list(inter_enc.parameters())+list(mlp_feature.parameters()),lr=args.lr)
    mae_fn=nn.L1Loss(reduction='mean')
    mse_fn=nn.MSELoss(reduction='mean')
    u_sens = torch.tensor(u_sens,dtype=torch.float).to(torch.long).to(args.device)
    e_su, e_si, _, _ = sens_enc.forward()
    e_su = e_su.detach().to(args.device)
    e_si = e_si.detach().to(args.device)
    best_perf = 0.0
    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    norm_adj_aug=inter_enc.norm_adj.clone().to(args.device)

    early_stop=0
    best_perf_log=None
    length_ori_0=sum(len(train_u2i[i]) for i in range(len(u_sens)) if u_sens[i] == 0)
    length_ori_1=sum(len(train_u2i[i]) for i in range(len(u_sens)) if u_sens[i] == 1)
    print(length_ori_0,length_ori_1)
    for epoch in range(args.num_epochs):
        mlp_feature.train()
        train_res = {
            'bpr': 0.0,
            'emb': 0.0,
            'ib': 0.0,
            'lb_aug':0.0
        }
        train_u2i_cp = train_u2i.copy()
        e_zu, e_zi = inter_enc.forward()
        u_g_embeddings, i_g_embeddings=inter_enc.propagate_all()# l outputs of l layer
        
        user_emb, item_emb = propagate(norm_adj_aug,e_zu,e_zi,e_su,e_si,n_users,n_items,u_g_embeddings, i_g_embeddings,mlp_feature)
        ## graph aug
        #e_bu=e_zu+(-mlp_graph(e_zu*e_su)).exp()*e_zu
        #e_bi=e_zi+(-mlp_graph(e_zi*e_si)).exp()*e_zi
        for user in range(n_users):
            item_list_pos = torch.tensor(train_u2i[user]).to(args.device)

            scores_1=F.sigmoid(torch.matmul(e_zu[user],e_zi[item_list_pos].t()))
            scores_2=F.sigmoid(torch.matmul(e_xu[user],(e_xi[item_list_pos]).t()))
            probability=((scores_1-torch.mean(scores_1))-(scores_2-torch.mean(scores_2))).exp()
            probability=torch.clamp(probability, min=0, max=1)
            probability_pyro=pyro.distributions.RelaxedBernoulliStraightThrough(temperature=1.0, probs=probability).rsample()
            #probability_pyro=torch.bernoulli(probability)
            train_u2i_cp[user] = item_list_pos[probability_pyro.bool()].cpu().tolist()
        
        length_0=sum(len(train_u2i_cp[i]) for i in range(len(u_sens)) if u_sens[i] == 0)
        length_1=sum(len(train_u2i_cp[i]) for i in range(len(u_sens)) if u_sens[i] == 1)
        
        with open('drop-ratio.csv', mode='a', newline='') as file:
            writer = csv.writer(file)

            writer.writerow([0, 1,2,3,4,5,6,7])

            writer.writerow([length_0, length_1, length_ori_0-length_0 ,length_ori_1-length_1 ,length_0/length_ori_0, length_1/length_ori_1,1-(length_0/length_ori_0), 1-(length_1/length_ori_1)])
            
            
        graph_aug = Graph(n_users, n_items, train_u2i_cp)
        norm_adj_aug = graph_aug.generate_ori_norm_adj().to(args.device)
        inter_enc_aug = LightGCN(n_users, n_items, norm_adj_aug, args.emb_size, args.n_layers, args.device)
        for uij in train_loader:
            u = uij[0].type(torch.long).to(args.device)
            i = uij[1].type(torch.long).to(args.device)
            j = uij[2].type(torch.long).to(args.device) 

            e_zu, e_zi = inter_enc.forward()
            u_g_embeddings, i_g_embeddings=inter_enc.propagate_all()
            #feature aug
            e_zu_aug, e_zi_aug = propagate(norm_adj_aug,e_zu,e_zi,e_su,e_si,n_users,n_items,u_g_embeddings, i_g_embeddings,mlp_feature)
            
            bpr_loss1, emb_loss = calc_bpr_loss(e_zu, e_zi, u, i, j)
            bpr_loss2, _ = calc_bpr_loss(e_zu_aug, e_zi_aug, u, i, j)
            
            # bpr both original and aug graphs, note that the same (u,i,j) used above
            bpr_loss=bpr_loss1+bpr_loss2*args.bpr_reg
            
            emb_loss = (emb_loss) * args.l2_reg

            lb_aug1=info_nce_for_embeddings(e_zu_aug[torch.unique(u)], e_zu[torch.unique(u)])
            lb_aug2=info_nce_for_embeddings(e_zi_aug[torch.unique(i)], e_zi[torch.unique(i)])
            lb_aug3=info_nce_for_embeddings(e_zu[torch.unique(u)], e_zu_aug[torch.unique(u)])
            lb_aug4=info_nce_for_embeddings(e_zi[torch.unique(i)], e_zi_aug[torch.unique(i)])
            lb_aug=(lb_aug1+lb_aug2+lb_aug3 + lb_aug4)*args.lareg
            
            #hsic
            up=calc_ib_loss(e_zu_aug[torch.unique(u)], e_su[torch.unique(u)], args.sigma)*args.ib_reg

            loss = bpr_loss +emb_loss+lb_aug+up

            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            train_res['ib'] += up.item()
            train_res['bpr'] += bpr_loss.item()
            train_res['emb'] += emb_loss.item()
            train_res['lb_aug'] += lb_aug.item()

        train_res['bpr'] = train_res['bpr'] / len(train_loader)
        train_res['emb'] = train_res['emb'] / len(train_loader)
        train_res['ib'] = train_res['ib'] / len(train_loader)
        train_res['lb_aug'] = train_res['lb_aug'] / len(train_loader)
        mlp_feature.eval()
        training_logs = 'epoch: %d, ' % epoch
        
        for name, value in train_res.items():
            training_logs += name + ':' + '%.6f' % value + ' '
        
        print(training_logs)
        
        early_stop+=1
        
        with torch.no_grad():
            t_user_emb, t_item_emb = inter_enc.forward()
            test_res = ranking_evaluate(
                user_emb=t_user_emb.detach().cpu().numpy(),
                item_emb=t_item_emb.detach().cpu().numpy(),
                n_users=n_users,
                n_items=n_items,
                train_u2i=train_u2i,
                test_u2i=test_u2i,
                sens=u_sens.cpu().numpy(),
                topks='[10,30]',
                num_workers=args.num_workers)
                

            p_eval = ''
            for keys, values in test_res.items():
                p_eval += keys + ':' + '[%.6f]' % values + ' '
            print(p_eval)
            if best_perf < test_res['ndcg@10']:
                early_stop=0
                best_perf = test_res['ndcg@10']
                best_perf_log=test_res
                torch.save(inter_enc, args.param_path)
                print('save successful')
            if early_stop>30:
                print('early_stop, best perf:')
                p_eval = ''
                for keys, values in best_perf_log.items():
                    p_eval += keys + ':' + '[%.6f]' % values + ' '
                print(p_eval)
                return
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ml_gcn_fairness',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    current_GMT = time.gmtime()
    torch.set_printoptions( precision=4,edgeitems=20,sci_mode=False,linewidth=160)

    time_stamp = calendar.timegm(current_GMT)
    parser.add_argument('--bakcbone', type=str, default='gcn')
    parser.add_argument('--dataset', type=str, default='./data/ml-1m/process/process.pkl')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_reg', type=float, default=0.001)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--log_path', type=str, default='logs/aug_ib/ml/ml_')
    parser.add_argument('--param_path', type=str, default='param/aug_ib/ml/ml_')
    parser.add_argument('--pretrain_path', type=str, default='param/gcn_base.pth')
    parser.add_argument('--bpr_reg', type=float, default=1.0)
    parser.add_argument('--lareg', type=float, default=0.1)
    parser.add_argument('--ib_reg', type=float, default=30.0)
    parser.add_argument('--sigma', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--sim_epochs', type=int, default=1000)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--train_step', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=2023)
    
    args = parser.parse_args()
    
    a = datetime.datetime.now()
    
    time_str = datetime.datetime.strftime(a, "%m-%d %H%M%S")
    
    pre_dex = "lareg=" + str(args.lareg) + "_ib_reg=" + str(args.ib_reg)+"_sigma=" + str(args.sigma)
    args.log_path = args.log_path + pre_dex + " " + time_str + ".txt"
    sys.stdout = Logger(args.log_path)
    args.param_path = args.param_path + pre_dex + " " + time_str + ".pth"
    #print('no graph aug and bpr')
    print(args)
    
    seed=args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    with open(args.dataset, 'rb') as f:
        train_u2i = pickle.load(f)
        train_i2u = pickle.load(f)
        test_u2i = pickle.load(f)
        test_i2u = pickle.load(f)
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        user_side_features = pickle.load(f)
        n_users, n_items = pickle.load(f)
    u_sens = user_side_features['gender']
    u_sens = np.array(u_sens, dtype=np.int64)
    dataset = BPRTrainLoader(train_set, train_u2i, n_items)
    graph = Graph(n_users, n_items, train_u2i)
    norm_adj = graph.generate_ori_norm_adj()
    classes_num=np.unique(u_sens).shape[0]
    sens_enc = SemiGCN(n_users, n_items, norm_adj,
                       args.emb_size, args.n_layers, args.device,
                       nb_classes=classes_num)
    ex_enc = torch.load(args.pretrain_path)
    e_xu, e_xi = ex_enc.forward()
    e_xu = e_xu.detach().to(args.device)
    e_xi = e_xi.detach().to(args.device)
    inter_enc = LightGCN(n_users, n_items, norm_adj, args.emb_size, args.n_layers, args.device)
    train_semigcn(sens_enc, u_sens, n_users,n_items, e_xu, e_xi, args,classes_num)
    train_unify_mi(sens_enc, inter_enc, e_xu, e_xi, dataset, u_sens, n_users,
                   n_items,train_u2i, test_u2i, args)
    sys.stdout = None