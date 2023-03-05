import copy
import datetime

import yaml
import time
import os
import sys
import torch
import logging
import pickle
import numpy as np
from scipy import sparse as sp
from yaml import Loader
import torch.nn as nn


# 读取配置文件
def read_cfg_file(filename):
    with open(filename, 'r') as yml_file:
        cfg = yaml.load(yml_file, Loader=Loader)
    return cfg


# 生成日志文件夹
def get_log_dir(kwargs):
    log_dir = kwargs['train'].get('log_dir')
    if log_dir is None:
        kwargs['data'].get('graph_pkl_filename')
        k = kwargs['model'].get('K')

        run_id = 'trans_gcn_k%s_%s/' % (
            k,
            time.strftime('%m%d%H%M%S'))
        base_dir = kwargs.get('base_dir')
        log_dir = os.path.join(base_dir, run_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


# 获取日志实体
def get_logger(log_dir,
               name,
               log_filename='info.log',
               level=logging.INFO,
               write_to_file=True):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    if write_to_file is True:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


# 获取训练设备
def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# 加载带有传感器id和邻接矩阵的图信息
def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


# 加载只带有邻接矩阵信息的图
def load_graph_data_adj_mx(pkl_filename):
    adj_mx = load_pickle(pkl_filename)
    return adj_mx.astype(np.float32)


# 加载pkl文件
def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


# 获取拉普拉斯位置编码
@DeprecationWarning
def laplacian_positional_encoding(adj, number_of_nodes, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = adj  # A为邻接矩阵
    N = np.diag(A.sum(axis=1)).clip(1) ** -0.5
    L = np.eye(number_of_nodes) - N * A * N

    # Eigenvectors with numpy
    eig_val, eig_vec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)
    idx = eig_val.argsort()  # increasing order
    eig_val, eig_vec = eig_val[idx], np.real(eig_vec[:, idx])
    lap_pos_enc = torch.from_numpy(eig_vec[:, 1:pos_enc_dim + 1]).float()
    return lap_pos_enc


@DeprecationWarning
def is_in_busy_time(deal_time):
    ymd = str(np.datetime64(deal_time)).split('T')[0]
    all_time = str(np.datetime64(deal_time)).split('.')[0]
    deal_time = all_time.replace('T', ' ')

    time_format = '%Y-%m-%d %H:%M:%S'
    deal_time = datetime.datetime.strptime(deal_time, time_format)
    # 范围时间
    time_zao_1 = datetime.datetime.strptime(ymd + ' 7:30:00', time_format)
    time_zao_2 = datetime.datetime.strptime(ymd + ' 9:30:00', time_format)

    time_wan_1 = datetime.datetime.strptime(ymd + ' 17:30:00', time_format)
    time_wan_2 = datetime.datetime.strptime(ymd + ' 19:30:00', time_format)

    # 判断当前时间是否在范围时间内
    if (time_zao_1 <= deal_time <= time_zao_2) or (time_wan_1 <= deal_time <= time_wan_2):
        return True
    else:
        return False


@DeprecationWarning
class PeakHighTroughLowLoss(nn.Module):
    def __init__(self):
        super(PeakHighTroughLowLoss, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')

    def forward(self, y_pred, y, y_time):
        high_time_x = list()
        high_time_y = list()
        trough_time_x = list()
        trough_time_y = list()

        for i, t in enumerate(y_time):
            if is_in_busy_time(t[0]):
                high_time_x.append(y_pred[i:i + 1, :, :, :])
                high_time_y.append(y[i:i + 1, :, :, :])
            else:
                trough_time_x.append(y_pred[i:i + 1, :, :, :])
                trough_time_y.append(y[i:i + 1, :, :, :])

        if len(high_time_x) > 0:
            high_time_x = torch.cat(high_time_x, dim=0)
            high_time_y = torch.cat(high_time_y, dim=0)
        if len(trough_time_x) > 0:
            trough_time_x = torch.cat(trough_time_x, dim=0)
            trough_time_y = torch.cat(trough_time_y, dim=0)

        if len(high_time_x) > 0 and len(trough_time_x) > 0:
            return 2 * self.l1_loss(high_time_x, high_time_y) + self.l1_loss(trough_time_x, trough_time_y)
        if len(high_time_x) > 0:
            return 2 * self.l1_loss(high_time_x, high_time_y)

        return self.l1_loss(trough_time_x, trough_time_y)


# 根据时间片动态获取相似度图
def get_time_index(first_time, category='train'):
    ymd = str(np.datetime64(first_time)).split('T')[0]
    d = ymd.split('-')[2]
    hms = str(np.datetime64(first_time)).split('T')[1]
    h_m_s = hms.split(':')

    index = 0
    if category == 'train':
        index += (int(d) - 1) * 66
    elif category == 'val':
        index += (int(d) - 19) * 66
    elif category == 'test':
        index += (int(d) - 21) * 66

    index += (int(h_m_s[0]) - 5) * 4
    index += (int(h_m_s[1])) // 15

    return index - 2


with open("data/hangzhou/dynamic_sim_35_train.pkl", 'rb') as f:
    dynamic_train_graph = pickle.load(f)
d_t_i = dynamic_train_graph['input']
d_t_o = dynamic_train_graph['output']

with open("data/hangzhou/dynamic_sim_35_val.pkl", 'rb') as f:
    dynamic_val_graph = pickle.load(f)

d_v_i = dynamic_val_graph['input']
d_v_o = dynamic_val_graph['output']

with open("data/hangzhou/dynamic_sim_35_test.pkl", 'rb') as f:
    dynamic_test_graph = pickle.load(f)

d_te_i = dynamic_test_graph['input']
d_te_o = dynamic_test_graph['output']

with open("data_deal/hz_correlation_all.pkl", 'rb') as f:
    od_every_day = pickle.load(f)

od_every_day = od_every_day


def get_multi_graph(graph_list, x_time, sim_rate, category='train'):
    edge_index_list = list()
    edge_attr_list = list()

    for xt in x_time:
        adj_mx_list_copy = copy.deepcopy(graph_list)
        sml_graph_index = get_time_index(xt[0], category)

        if category == 'train':
            input_i = np.array(d_t_i[sml_graph_index])
            output_i = np.array(d_t_o[sml_graph_index])
        elif category == 'val':
            input_i = np.array(d_v_i[sml_graph_index])
            output_i = np.array(d_v_o[sml_graph_index])
        elif category == 'test':
            input_i = np.array(d_te_i[sml_graph_index])
            output_i = np.array(d_te_o[sml_graph_index])

        # adj_mx_list_copy.append(input_i + output_i)
        adj_mx_list_copy[1] += (input_i + output_i) * sim_rate

        ymd = str(np.datetime64(xt[0])).split('T')[0]
        d = int(ymd.split('-')[2])

        adj_mx_list_copy[2] += np.array(od_every_day[d - 1]).astype(np.float32) * 0.5

        adj_mx = np.stack(adj_mx_list_copy, axis=-1)

        adj_mx = adj_mx / (adj_mx.sum(axis=0) + 1e-18)

        src, dst = adj_mx.sum(axis=-1).nonzero()
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=get_device())
        edge_attr = torch.tensor(adj_mx[adj_mx.sum(axis=-1) != 0],
                                 dtype=torch.float,
                                 device=get_device())
        edge_index_list.append(edge_index)
        edge_attr_list.append(edge_attr)

    return edge_index_list, edge_attr_list
