import torch
import torch.nn as nn
from utils import process
from layers import GCN, AvgNeighbor, Discriminator
import torch.nn.functional as F
import sklearn.cluster
import numpy as np

class WeakAD(nn.Module):
    def __init__(self, fea_hid, gcn_hid, trans_hid, node_num, device):
        super(WeakAD, self).__init__()
        self.device = device
        self.node_num = node_num
        self.gcn = GCN(fea_hid, gcn_hid)
        self.gcn2 = GCN(gcn_hid, gcn_hid)
        self.transq = nn.Linear(gcn_hid, trans_hid, bias=True)
        self.score = nn.Linear(trans_hid, 1, bias=True)
        self.act = nn.PReLU()

        self.avg_neighbor = AvgNeighbor()
        self.disc1 = Discriminator(fea_hid, trans_hid)
        self.disc2 = Discriminator(trans_hid, gcn_hid)


    def forward(self, adj, adj_avg, feat, neg_num, a_subg_list, n_subg_list, a_subg_lap, n_subg_lap):
        
        node_vec, _, node_transemb = self.gcn(feat, adj)
        node_vec, _, _ = self.gcn2(node_vec, adj)

        node_vec = self.act(self.transq(node_vec))

        node_neighbor = self.avg_neighbor(node_transemb, adj_avg)

        res_mi_pos, res_mi_neg = self.disc1(node_vec, feat, process.negative_sampling(self.node_num, neg_num))
        res_local_pos, res_local_neg = self.disc2(node_neighbor, node_vec, process.negative_sampling(self.node_num, neg_num))

        node_score = torch.sigmoid(self.score(node_vec))
        # print ('node score: ', node_score.flatten())

        ############# refined score & mil ranking loss & sparsity loss & homophily loss ##############

        ranking_loss = torch.tensor(0.0).to(self.device)
        sp_loss = torch.tensor(0.0).to(self.device)
        hom_loss = torch.tensor(0.0).to(self.device)
        ranking_inner_loss = torch.tensor(0.0).to(self.device)

        
        for i in range(len(a_subg_list)):
            a_idxs = a_subg_list[i]
            a_mil_score = node_score[a_idxs].flatten()
            hom_loss = hom_loss+torch.mm(a_mil_score.unsqueeze(0), torch.spmm(a_subg_lap[i], a_mil_score.unsqueeze(1))).squeeze()
            a_mil_score_sort, _ = torch.sort(a_mil_score, descending=True)
            a_mil_avg_idx = torch.where(a_mil_score_sort<torch.mean(a_mil_score_sort))[0][0]

            n_idxs = n_subg_list[i]
            n_mil_score = node_score[n_idxs].flatten()
            hom_loss = hom_loss+torch.mm(n_mil_score.unsqueeze(0), torch.spmm(n_subg_lap[i], n_mil_score.unsqueeze(1))).squeeze()
            n_mil_score_sort, _ = torch.sort(n_mil_score, descending=True)

            ranking_loss = ranking_loss+torch.relu(1.0-a_mil_score_sort[0]+n_mil_score_sort[0])
            ranking_inner_loss = ranking_inner_loss+torch.relu(1.0-a_mil_score_sort[0]+a_mil_score_sort[a_mil_avg_idx])
            
            sp_loss = sp_loss+torch.sum(a_mil_score)

        ranking_loss = ranking_loss/len(a_subg_list)
        sp_loss = sp_loss/len(a_subg_list)
        hom_loss = hom_loss/(len(a_subg_list)*2)
        ranking_inner_loss = ranking_inner_loss/len(a_subg_list)

        ############# reconstruction loss ##############

        adj_rebuilt = torch.sigmoid(torch.mm(node_vec, torch.t(node_vec)))

        return ranking_loss, ranking_inner_loss, sp_loss, hom_loss, adj_rebuilt, res_mi_pos, res_mi_neg, res_local_pos, res_local_neg

    def get_refiend_score(self, adj, feat):

        node_vec, _, _ = self.gcn(feat, adj)
        node_vec, _, _ = self.gcn2(node_vec, adj)

        node_vec = self.act(self.transq(node_vec))

        node_score = torch.sigmoid(self.score(node_vec))

        return node_score.flatten().detach()
