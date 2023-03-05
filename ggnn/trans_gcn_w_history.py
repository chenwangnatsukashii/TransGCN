import math

from torch import nn
import torch
from torch.nn import Parameter as Param

import numpy as np
from ggnn.rgcn import RGCNConv
from data.data_all import get_all_date


class PositionWiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super(PositionWiseFeedforwardLayer, self).__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)

        return x


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, n_nodes, batch_size):
        super(MultiHeadAttentionLayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.n_nodes = n_nodes
        self.head_dim = hid_dim // n_heads
        self.batch_size = batch_size

        self.Wqs = Param(torch.Tensor(hid_dim, hid_dim))
        self.Wks = Param(torch.Tensor(hid_dim, hid_dim))
        self.Wvs = Param(torch.Tensor(hid_dim, hid_dim))

        torch.nn.init.xavier_uniform_(self.Wqs)
        torch.nn.init.xavier_uniform_(self.Wks)
        torch.nn.init.xavier_uniform_(self.Wvs)

        # self.query_projection = nn.Linear(hid_dim, hid_dim)
        # self.key_projection = nn.Linear(hid_dim, hid_dim)
        # self.value_projection = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        # self.conv_q = nn.Conv1d(in_channels=640, out_channels=640, kernel_size=3, stride=1, padding=1)
        # self.conv_k = nn.Conv1d(in_channels=640, out_channels=640, kernel_size=3, stride=1, padding=1)
        # self.conv_v = nn.Conv1d(in_channels=640, out_channels=640, kernel_size=1, stride=1, padding=1)

    def forward(self, query, key, value):
        q_h = torch.matmul(query, self.Wqs)
        k_h = torch.matmul(key, self.Wks)
        v_h = torch.matmul(value, self.Wvs)

        q_step = query.size(0)
        k_step = key.size(0)
        v_step = value.size(0)

        # queries = self.query_projection(query).view(q_step, self.n_nodes * self.batch_size, self.n_heads, self.head_dim)
        # keys = self.key_projection(key).view(k_step, self.n_nodes * self.batch_size, self.n_heads, self.head_dim)
        # values = self.value_projection(value).view(v_step, self.n_nodes * self.batch_size, self.n_heads, self.head_dim)

        # q_h = self.conv_q(query)
        # k_h = self.conv_k(key)
        # v_h = torch.matmul(value, self.Wvs)

        queries = q_h.view(q_step, self.n_nodes * self.batch_size, self.n_heads, self.head_dim)
        keys = k_h.view(k_step, self.n_nodes * self.batch_size, self.n_heads, self.head_dim)
        values = v_h.view(v_step, self.n_nodes * self.batch_size, self.n_heads, self.head_dim)

        scores = queries * keys
        result = torch.softmax(scores / math.sqrt(self.hid_dim), dim=-1)

        return self.fc((values * result).view(-1, self.n_nodes * self.batch_size, self.hid_dim))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, pf_dim, n_nodes, batch_size, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttentionLayer(d_model, n_heads, n_nodes, batch_size)
        self.position_wise_feed_forward = PositionWiseFeedforwardLayer(d_model, pf_dim, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=pf_dim, kernel_size=(1, 1))
        self.conv2 = nn.Conv1d(in_channels=pf_dim, out_channels=d_model, kernel_size=(1, 1))

    def forward(self, src):
        _src = self.self_attention(src, src, src)
        src = src + self.dropout(_src)

        # src = self.norm1(src)
        # _src = self.position_wise_feed_forward(src)
        # src = src + self.dropout(_src)
        # src = self.norm2(src)
        # return src

        y = x = self.norm1(src)
        y = torch.unsqueeze(y.transpose(-1, 1), dim=-1)
        y = self.dropout(torch.relu(self.conv1(y)))
        y = torch.squeeze(self.dropout(self.conv2(y)), dim=-1).transpose(-1, 1)
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, pf_dim, n_nodes, batch_size, dropout):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([EncoderLayer(d_model,
                                                  n_heads,
                                                  pf_dim,
                                                  n_nodes,
                                                  batch_size,
                                                  dropout)
                                     for _ in range(n_layers)])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)

        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, pf_dim, n_nodes, batch_size, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttentionLayer(d_model, n_heads, n_nodes, batch_size)
        self.encoder_attention = MultiHeadAttentionLayer(d_model, n_heads, n_nodes, batch_size)

        self.position_wise_feed_forward = PositionWiseFeedforwardLayer(d_model,
                                                                       pf_dim,
                                                                       dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=pf_dim, kernel_size=(1, 1))
        self.conv2 = nn.Conv1d(in_channels=pf_dim, out_channels=d_model, kernel_size=(1, 1))

    def forward(self, tgt, enc_src):
        tgt2 = self.self_attention(tgt, tgt, tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.encoder_attention(tgt, enc_src, enc_src)
        tgt = tgt + self.dropout(tgt2)

        # tgt = self.norm2(tgt)
        # tgt2 = self.position_wise_feed_forward(tgt)
        # tgt = tgt + self.dropout(tgt2)
        # tgt = self.norm3(tgt)
        # return tgt

        y = x = self.norm2(tgt)
        y = torch.unsqueeze(y.transpose(-1, 1), dim=-1)
        y = self.dropout(torch.relu(self.conv1(y)))
        y = torch.squeeze(self.dropout(self.conv2(y)), dim=-1).transpose(-1, 1)
        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, hid_dim, n_layers, n_heads, pf_dim, n_nodes, batch_size, dropout):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList(
            [DecoderLayer(hid_dim, n_heads, pf_dim, n_nodes, batch_size, dropout) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_outputs):
        for layer in self.layers:
            dec_inputs = layer(dec_inputs, enc_outputs)

        return torch.squeeze(dec_inputs, dim=0)


class KStepRGCN(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            num_relations,
            num_bases,
            K,
            bias,
    ):
        super(KStepRGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.K = K
        self.rgcn_layers = nn.ModuleList([
                                             RGCNConv(in_channels,
                                                      out_channels,
                                                      num_relations,
                                                      num_bases,
                                                      bias)
                                         ] + [
                                             RGCNConv(out_channels,
                                                      out_channels,
                                                      num_relations,
                                                      num_bases,
                                                      bias) for _ in range(self.K - 1)
                                         ])

    def forward(self, x, edge_index, edge_attr):
        for i in range(self.K):
            x = self.rgcn_layers[i](x=x,
                                    edge_index=edge_index,
                                    edge_attr=edge_attr,
                                    edge_norm=None)
            # not final layer, add relu
            if i != self.K - 1:
                x = torch.relu(x)

        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # return self.pe[:, :x.size(1)]
        return self.pe[x]


def init_position_embedding(max_len, d_model):
    pe = torch.zeros(max_len, d_model).float()
    position = torch.arange(0.0, max_len).unsqueeze(1)
    div_term = 1 / np.power(10000, (torch.arange(0, d_model, 2) // d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


pe = init_position_embedding(24 * 4, 256)


def get_time_position_encoding(index, n_nodes):
    index = int(index)

    return pe[index].repeat(n_nodes, 1)


class TSESC(nn.Module):
    def __init__(self, cfg):
        super(TSESC, self).__init__()
        self.in_dim = cfg['model']['input_dim']
        self.out_dim = cfg['model']['output_dim']
        self.pf_dim = self.out_dim * 4
        self.n_heads = cfg['model']['n_heads']
        self.num_nodes = cfg['model']['num_nodes']
        self.batch_size = cfg['data']['batch_size']
        self.horizon = cfg['model']['horizon']
        self.dropout_prob = cfg['model']['dropout_prob']
        self.en_layers = cfg['model']['en_layers']
        self.de_layers = cfg['model']['de_layers']
        self.num_relations = cfg['model']['num_relations']
        self.num_bases = cfg['model']['num_bases']
        self.k = cfg['model']['K']

        self.encoder_gcn = nn.ModuleList([KStepRGCN(self.in_dim,
                                                    self.out_dim,
                                                    num_relations=self.num_relations,
                                                    num_bases=self.num_bases,
                                                    K=self.k,
                                                    bias=False) for _ in range(self.horizon)])

        self.decoder_gcn = nn.ModuleList([KStepRGCN(self.in_dim,
                                                    self.out_dim,
                                                    num_relations=self.num_relations,
                                                    num_bases=self.num_bases,
                                                    K=self.k,
                                                    bias=False) for _ in range(self.horizon)])

        self.encoder = Encoder(self.out_dim, self.en_layers, self.n_heads, self.pf_dim, self.num_nodes, self.batch_size,
                               self.dropout_prob)

        self.decoder = Decoder(self.out_dim, self.de_layers, self.n_heads, self.pf_dim, self.num_nodes, self.batch_size,
                               self.dropout_prob)

        self.output_layer = nn.Linear(self.out_dim, self.in_dim)

        self.temporal_list = list()
        for i in range(96):
            self.temporal_list.append(Param(get_time_position_encoding(i, self.num_nodes)))

        self.position_embedding = PositionalEmbedding(24 * 4, self.out_dim)

    def forward(self, sequences, scaler):
        edge_index = sequences[0].edge_index.detach()
        edge_attr = sequences[0].edge_attr.detach()

        cur_input_list = list()
        xtime_list = list()
        ytime_list = list()

        for ii, batch in enumerate(sequences):
            for i in range(self.batch_size):
                xtime = str(np.datetime64(batch.xtime[i])).split('T')[1]
                hour = int(xtime.split(':')[0])
                minute = int(xtime.split(':')[1])
                # xtime_list.append(self.temporal_list[hour * 4 + minute // 15 - 1])
                # ytime_list.append(self.temporal_list[hour * 4 + minute // 15 + 3])

                xtime_list.append(self.position_embedding(hour * 4 + minute // 15 - 1).repeat(self.num_nodes, 1))
                ytime_list.append(self.position_embedding(hour * 4 + minute // 15 + 3).repeat(self.num_nodes, 1))

            cur_input_list.insert(ii, batch.x)
            cur_input_list.insert(ii + self.horizon, batch.y)

        input_gcn_result = list()
        for i, xs in enumerate(cur_input_list[:self.horizon]):
            xs_i = self.encoder_gcn[i](x=xs, edge_index=edge_index, edge_attr=edge_attr)
            input_gcn_result.append(
                xs_i + torch.cat(xtime_list[i * self.batch_size:(i + 1) * self.batch_size]).to(torch.device('cuda')))

        enc_inputs = torch.stack(input_gcn_result)

        enc_outputs = self.encoder(enc_inputs)

        history_y = list()
        for ii in sequences[0].ytime:
            history_y.append(get_all_date(ii))

        history_y = scaler.transform(history_y)
        xs = torch.tensor(np.concatenate(history_y, axis=1), dtype=torch.float32, device=torch.device('cuda'))

        input_gcn_decoder = list()

        for i in range(self.horizon):
            xs_o = self.decoder_gcn[i](x=torch.squeeze(xs[i:i + 1, :, :], dim=0), edge_index=edge_index,
                                       edge_attr=edge_attr)
            input_gcn_decoder.append(
                xs_o + torch.cat(ytime_list[i * self.batch_size:(i + 1) * self.batch_size]).to(torch.device('cuda')))

        dec_inputs = torch.stack(input_gcn_decoder)
        dec_out = self.decoder(dec_inputs, enc_outputs)
        return self.output_layer(dec_out)
