import torch
import torch.nn as nn
from utils import process

# Applies mean-pooling on neighbors
class AvgNeighbor(nn.Module):
    def __init__(self):
        super(AvgNeighbor, self).__init__()
        self.act = nn.PReLU()

    def forward(self, seq, adj_avg):

        return self.act(torch.spmm(adj_avg, seq))
