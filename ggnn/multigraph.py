import torch
import math
import sys
import os
from ggnn.trans_gcn import TransGCN

sys.path.insert(0, os.path.abspath('..'))


class Net(torch.nn.Module):

    def __init__(self, cfg):
        super(Net, self).__init__()
        self.num_nodes = cfg['model']['num_nodes']
        self.input_dim = cfg['model']['input_dim']
        self.horizon = cfg['model']['horizon']
        self.type = cfg['model']['type']
        self.batch_size = cfg['data']['batch_size']

        self.trans_gcn = TransGCN(cfg)

    @staticmethod
    def _compute_sampling_threshold(step, k):
        return k / (k + math.exp(step / k))

    def forward(self, sequences, scaler):
        output = self.trans_gcn(sequences, scaler)
        return output.view(self.horizon, self.batch_size, self.num_nodes, self.input_dim).permute(1, 0, 2, 3)
