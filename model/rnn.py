import torch
import torch.nn as nn
from model.tools import *

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, depth):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth

        self.W_i = nn.Sequential( nn.Linear(input_size + hidden_size, hidden_size), nn.Sigmoid() )
        self.W_o = nn.Sequential( nn.Linear(input_size + hidden_size, hidden_size), nn.Sigmoid() )
        self.W_f = nn.Sequential( nn.Linear(input_size + hidden_size, hidden_size), nn.Sigmoid() )
        self.W = nn.Sequential( nn.Linear(input_size + hidden_size, hidden_size), nn.Tanh() )

    def get_init_state(self, fmess, init_state=None):
        h = torch.zeros(len(fmess), self.hidden_size, device=fmess.device)
        c = torch.zeros(len(fmess), self.hidden_size, device=fmess.device)
        if init_state is not None:
            h = torch.cat( (h, init_state), dim=0)
            c = torch.cat( (c, torch.zeros_like(init_state)), dim=0)
        return h,c

    def get_hidden_state(self, h):
        return h[0]

    def LSTM(self, x, h_nei, c_nei):
        h_sum_nei = h_nei.sum(dim=1)
        x_expand = x.unsqueeze(1).expand(-1, h_nei.size(1), -1)
        i = self.W_i( torch.cat([x, h_sum_nei], dim=-1) )
        o = self.W_o( torch.cat([x, h_sum_nei], dim=-1) )
        f = self.W_f( torch.cat([x_expand, h_nei], dim=-1) )
        u = self.W( torch.cat([x, h_sum_nei], dim=-1) )
        c = i * u + (f * c_nei).sum(dim=1)
        h = o * torch.tanh(c)
        return h, c

    def forward(self, fmess, bgraph):
        h = torch.zeros(fmess.size(0), self.hidden_size, device=fmess.device)
        c = torch.zeros(fmess.size(0), self.hidden_size, device=fmess.device)
        mask = torch.ones(h.size(0), 1, device=h.device)
        mask[0, 0] = 0 #first message is padding

        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            c_nei = index_select_ND(c, 0, bgraph)
            h,c = self.LSTM(fmess, h_nei, c_nei)
            h = h * mask
            c = c * mask
        return h,c

    def sparse_forward(self, h, fmess, submess, bgraph):
        h,c = h
        mask = h.new_ones(h.size(0)).scatter_(0, submess, 0)
        h = h * mask.unsqueeze(1)
        c = c * mask.unsqueeze(1)
        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            c_nei = index_select_ND(c, 0, bgraph)
            sub_h, sub_c = self.LSTM(fmess, h_nei, c_nei)
            h = index_scatter(sub_h, h, submess)
            c = index_scatter(sub_c, c, submess)
        return h,c



