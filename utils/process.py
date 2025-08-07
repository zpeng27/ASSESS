import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import pickle
import random
import walker
import scipy.io as scio
from sklearn import preprocessing
from scipy.sparse.csgraph import laplacian
from sklearn.metrics import roc_auc_score, average_precision_score

###############################################
# This section of code adapted from tkipf/GCN and Petar Veličković/DGI #
###############################################

def evaluation(gnd, score):
    
    auc = roc_auc_score(gnd, score)*100
    print ('ROC-AUC: ', auc)

    ap = average_precision_score(gnd, score)*100
    print ('AP: ', ap)

    return auc, ap


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def class_to_onehot(class_array):
    max_class = np.max(class_array)
    min_class = np.min(class_array)
    # print ('Class MAX: ', max_class)
    # print ('Class MIN: ', min_class)
    class_num = max_class - min_class +1
    class_onehot = np.eye(class_array.shape[0])

    gnd_onehot = class_onehot[class_array][:, range(class_num)]
    return gnd_onehot


def gen_trainset(device, adj, feature, gnd_node_list, walks, unlabel_idx_pool, label_idx_pool, number_label_gnd_center):
    unlabel_node_list, nodeidx_protupdate, unlabel_subg_lapmatrix = get_subg_norm(device, adj, walks, unlabel_idx_pool, label_idx_pool, number_label_gnd_center)

    gnd_node_list = [np.array(i).astype(np.int32) for i in gnd_node_list]
    unlabel_node_list = [np.array(i).astype(np.int32) for i in unlabel_node_list]
    nodeidx_protupdate = np.array(nodeidx_protupdate).astype(np.int32)

    return gnd_node_list, unlabel_node_list, nodeidx_protupdate, unlabel_subg_lapmatrix


def get_node_cls_weight(attn, attn_reverse, gnd_onehot, gnd_onehot_reverse):
    node_cls_logits = torch.mul(attn, gnd_onehot) + torch.mul(attn_reverse, gnd_onehot_reverse)
    return node_cls_logits

def get_subg_norm(device, adj, walks, unlabel_idx_pool, label_idx_pool, number_label_gnd_center):
    selected_unlabel_center = np.random.choice(unlabel_idx_pool, number_label_gnd_center)
    unlabel_subgraph_idx = walks[selected_unlabel_center]

    unlabel_subg_size = []
    unlabel_subg_list = []
    unlabel_subg_nodeidx_protupdate = []
    unlabel_subg_lapmatrix = []
    label_idx_pool = set(label_idx_pool)
    for i in unlabel_subgraph_idx:
        tmp = list(set(i).difference(label_idx_pool))
        tmp.sort()
        l_matrix = laplacian(adj[tmp,:][:,tmp], normed=False)
        l_matrix_tensor = sparse_mx_to_torch_sparse_tensor(l_matrix)
        l_matrix_tensor = l_matrix_tensor.to(device)
        unlabel_subg_lapmatrix.append(l_matrix_tensor)
        unlabel_subg_list.append(tmp)
        unlabel_subg_size.append(len(tmp))
        unlabel_subg_nodeidx_protupdate.extend(tmp)

    print ('unlabel/norm subg size: ', unlabel_subg_size, '\navg size: ', np.mean(np.array(unlabel_subg_size)))

    return unlabel_subg_list, list(set(unlabel_subg_nodeidx_protupdate)), unlabel_subg_lapmatrix

def get_subg_abnorm(adj, gnd, number_label_gnd_center, dataset_file, device):
    gnd_idx = np.where(gnd!=0)[0]
    #graph = nx.from_numpy_matrix(adj.A)
    graph = nx.from_numpy_array(adj.A)

    walks_path = dataset_file.rsplit('.', 1)[0]+'_walks.npy'
    if os.path.exists(walks_path):
        walks = np.load(walks_path)
    else:
        walks = walker.random_walks(graph, n_walks=1, walk_len=15, alpha=.2) # ndarray
        np.save(walks_path, walks)

    rng = np.random.RandomState(123)
    selected_gnd_center = rng.choice(gnd_idx, number_label_gnd_center)
    gnd_subgraph_idx = walks[selected_gnd_center]

    gnd_subgraph_idx_flat = gnd_subgraph_idx.flatten()
    label_idx_pool = list(set(gnd_subgraph_idx_flat))
    node_idx_all = list(np.arange(graph.number_of_nodes()))
    unlabel_idx_pool = list(set(node_idx_all) - set(label_idx_pool))

    gnd_subg_size = []
    gnd_subg_list = []
    gnd_subg_lapmatrix = []
    for i in gnd_subgraph_idx:
        tmp = list(set(i))
        tmp.sort()
        l_matrix = laplacian(adj[tmp,:][:,tmp], normed=False)
        l_matrix_tensor = sparse_mx_to_torch_sparse_tensor(l_matrix)
        l_matrix_tensor = l_matrix_tensor.to(device)
        gnd_subg_lapmatrix.append(l_matrix_tensor)
        gnd_subg_list.append(tmp)
        gnd_subg_size.append(len(tmp))

    print ('gnd subg size: ', gnd_subg_size, '\navg size: ', np.mean(np.array(gnd_subg_size)))

    return gnd_subg_list, unlabel_idx_pool, label_idx_pool, walks, gnd_subg_lapmatrix

def load_data(dataset_file, number_label_gnd_center, device):
    """Load data."""
    
    with open(dataset_file, 'rb') as f:
        data = pickle.load(f)
        adj = build_symmetric(data['A'])
        feature = data['X']
        gnd = data['gnd']

    l_matrix = laplacian(adj, normed=False)

    gnd_subg_list, unlabel_idx_pool, label_idx_pool, walks, gnd_subg_lapmatrix = get_subg_abnorm(adj, gnd, number_label_gnd_center, dataset_file, device)

    return adj, feature, gnd, gnd_subg_list, unlabel_idx_pool, label_idx_pool, walks, l_matrix, gnd_subg_lapmatrix

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx

def preprocess_features(features):
    features = features.A
    print ('raw feat')
    print (features)
    print (np.max(features))
    print (np.min(features))
    features = features*10  #you could change the preprocessing strategy according to the dataset
    #features = preprocessing.normalize(features, norm='l2', axis=1)
    print ('preprocessed feat')
    print (features)
    print (np.max(features))
    print (np.min(features))
    return features

def build_symmetric(adj):
    adj = adj.tocoo()
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj.tocsr()

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def avg_adj(adj):
    """row average adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_avg = np.power(rowsum, -1.0).flatten()
    d_inv_avg[np.isinf(d_inv_avg)] = 0.
    d_inv_avg[np.isnan(d_inv_avg)] = 0.
    d_mat_inv_avg = sp.diags(d_inv_avg)
    
    return adj.dot(d_mat_inv_avg).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def reconstruct_loss(pre, gnd, weight1, weight2):
    temp1 = gnd*torch.log(pre+(1e-10))*(-weight1)
    temp2 = (1-gnd)*torch.log(1-pre+(1e-10))
    return torch.mean(temp1-temp2)*weight2

def make_modularity_matrix(adj):
    adj = adj*(torch.ones(adj.shape[0], adj.shape[0]) - torch.eye(adj.shape[0]))
    degrees = adj.sum(dim=0).unsqueeze(1)
    mod = adj - degrees@degrees.t()/adj.sum()
    return mod

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def negative_sampling(node_num, sample_times):
    sample_list = []
    for j in range(sample_times):
        sample_iter = []
        i = 0
        while True:
            randnum = np.random.randint(0,node_num)
            if randnum!=i:
                sample_iter.append(randnum)
                i = i+1
            if len(sample_iter)==node_num:
                break
        sample_list.append(sample_iter)
    return sample_list

def sp_func(arg):
    return torch.log(1+torch.exp(arg))

def mi_loss_jsd(pos, neg):
    e_pos = torch.mean(sp_func(-pos))
    e_neg = torch.mean(torch.mean(sp_func(neg),0))
    return e_pos+e_neg
