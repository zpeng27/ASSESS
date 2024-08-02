import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_h1, n_h2):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h1, n_h2, 1)
        self.act = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, h_c, h_pl, sample_list):
        sc_1 = torch.unsqueeze(torch.squeeze(self.act(self.f_k(h_pl, h_c))), dim=0)
        
        sc_2_list = []
        for i in range(len(sample_list)):
            h_mi = h_pl[sample_list[i]]
            sc_2_list.append(torch.squeeze(self.act(self.f_k(h_mi, h_c))))
        sc_2 = torch.stack(sc_2_list)

        return sc_1, sc_2

