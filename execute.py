import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import scipy.sparse as sp
from models import WeakAD
from utils import process
import scipy
import random
import math
import torch.nn.functional as F

"""command-line interface"""
parser = argparse.ArgumentParser(description="PyTorch Implementation of WeakAD")
parser.add_argument('--dataset_file', default='./data/reddit.pkl', help='name of dataset')
parser.add_argument('--cuda', type=str, default='cuda:0', help='cuda/cpu')
"""training params"""
parser.add_argument('--gcn_hid', type=int, default=32,
                    help='dim of node embedding (default: 512)')
parser.add_argument('--trans_hid', type=int, default=28,
                    help='dim of node embedding (default: 512)')
parser.add_argument('--nb_epochs', type=int, default=10000,
                    help='number of epochs to train (default: 550)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size for gnd/unlabel (default: 8)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--l2_coef', type=float, default=5e-4,
                    help='weight decay (default: 0.0)')
parser.add_argument('--negative_num', type=int, default=5,
                    help='number of negative examples used in the discriminator (default: 5)')
parser.add_argument('--number_label_gnd_center', type=int, default=50,
                    help='number of labeled abnormal subgraphs or normal subgraphs')
parser.add_argument('--alpha', type=float, default=0.1,
                    help='parameter for sparse loss')
parser.add_argument('--beta', type=float, default=0.8,
                    help='parameter for hom loss')
parser.add_argument('--gamma', type=float, default=5.0,
                    help='parameter for ranking loss')

###############################################
# This section of code adapted from Petar Veličković/DGI #
###############################################

args = parser.parse_args()
device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')

print('Loading ', args.dataset_file)
input_adj, input_features, node_gnd, gnd_subg_list, unlabel_idx_pool, label_idx_pool, walks, l_matrix, gnd_subg_lapmatrix = process.load_data(args.dataset_file, args.number_label_gnd_center, device)

nb_nodes = input_features.shape[0]
ft_size = input_features.shape[1]


adj_norm = process.normalize_adj(input_adj + sp.eye(nb_nodes))
adj_tensor = process.sparse_mx_to_torch_sparse_tensor(adj_norm)
features = process.preprocess_features(input_features)
features_tensor = torch.FloatTensor(features)

adj_avg = process.avg_adj(input_adj)
adj_avg_tensor = process.sparse_mx_to_torch_sparse_tensor(adj_avg)

adj_target = input_adj.A+np.eye(nb_nodes)
edges_n = np.sum(adj_target)/2
weight1 = (nb_nodes*nb_nodes-edges_n)*1.0/edges_n
weight2 = nb_nodes*nb_nodes*1.0/(nb_nodes*nb_nodes-edges_n)
adj_target = torch.FloatTensor(adj_target)


model = WeakAD(ft_size, args.gcn_hid, args.trans_hid, nb_nodes, device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
 
model.to(device)
adj_tensor = adj_tensor.to(device)
features_tensor = features_tensor.to(device)
adj_target = adj_target.to(device)
adj_avg_tensor = adj_avg_tensor.to(device)


best_auc = 0
best_ap = 0
a_idxs, n_idxs, n_idxs_protupdate, unlabel_subg_lapmatrix = process.gen_trainset(device, input_adj, features, gnd_subg_list, walks, unlabel_idx_pool, label_idx_pool, args.number_label_gnd_center)

# torch.autograd.set_detect_anomaly(True)

for epoch in range(args.nb_epochs):
    print ('=============== Epoch:', (epoch+1), ' ===============')

    for batch_idx in range(math.ceil(args.number_label_gnd_center/args.batch_size)):
        
        model.train()
        optimizer.zero_grad()
        
        end_idx = (batch_idx+1)*args.batch_size
        if (end_idx>args.number_label_gnd_center):
            end_idx = args.number_label_gnd_center
        
        ranking_loss, sp_loss, hom_loss, adj_rebuilt, res_mi_pos, res_mi_neg, res_local_pos, res_local_neg = model(adj_tensor, adj_avg_tensor, features_tensor, args.negative_num, a_idxs[batch_idx*args.batch_size:end_idx], n_idxs[batch_idx*args.batch_size:end_idx], gnd_subg_lapmatrix[batch_idx*args.batch_size:end_idx], unlabel_subg_lapmatrix[batch_idx*args.batch_size:end_idx])

        adj_mi_loss = process.reconstruct_loss(adj_rebuilt, adj_target, weight1, weight2)
        feat_mi_loss = process.mi_loss_jsd(res_mi_pos, res_mi_neg) + process.mi_loss_jsd(res_local_pos, res_local_neg)

        loss = args.gamma*ranking_loss+args.alpha*sp_loss+args.beta*hom_loss+adj_mi_loss+feat_mi_loss
        
        print('Batch: %.3d'%(batch_idx+1), 'Loss: %.8f'%loss.item(), 'ranking Loss: %.8f'%ranking_loss.item(), 'sp Loss: %.8f'%sp_loss.item(), \
                'hom Loss: %.8f'%hom_loss.item(), 'adj Loss: %.8f'%adj_mi_loss.item(), 'feat Loss: %.8f'%feat_mi_loss.item())

        loss.backward()
        optimizer.step()

    if (epoch+1)%5 ==0:
        model.eval()
        prot_abnorm_sim = model.get_refiend_score(adj_tensor, features_tensor)
        auc, ap = process.evaluation(node_gnd, prot_abnorm_sim.cpu().numpy())
        
        if best_auc<auc:
            best_auc = auc
        if best_ap<ap:
            best_ap = ap
            
        print ('Best AUC: ', best_auc)
        print ('Best AP: ', best_ap)