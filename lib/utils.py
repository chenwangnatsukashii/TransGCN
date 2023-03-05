import logging
import os
import sys
import numpy as np
import torch
from torch_geometric.data import Batch, Data
import scipy.sparse as sp
from trans_utils import load_pickle


class DataLoader(object):

    def __init__(self,
                 xs,
                 ys,
                 xtime,
                 ytime,
                 batch_size,
                 pad_with_last_sample=True,
                 shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            xtime_padding = np.repeat(xtime[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            ytime_padding = np.repeat(ytime[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            xtime = np.concatenate([xtime, xtime_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            ytime = np.concatenate([ytime, ytime_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
            xtime, ytime = xtime[permutation], ytime[permutation]
        self.xs = xs
        self.ys = ys
        self.xtime = xtime
        self.ytime = ytime

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size,
                              self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                xtime_i = self.xtime[start_ind:end_ind, ...]
                ytime_i = self.ytime[start_ind:end_ind, ...]
                yield (x_i, y_i, xtime_i, ytime_i)
                self.current_ind += 1

        return _wrapper()


class StandardScalerTorch:
    """
    Standard the input
    """

    def __init__(self, mean, std, device):
        self.mean = torch.tensor(data=mean, dtype=torch.float, device=device)
        self.std = torch.tensor(data=std, dtype=torch.float, device=device)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def load_dataset(dataset_dir, batch_size, test_batch_size=None, **kwargs):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][...,
                                                 0].mean(),
                            std=data['x_train'][...,
                                                0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][...,
                              0] = scaler.transform(data['x_' + category][...,
                                                                          0])
        data['y_' + category][...,
                              0] = scaler.transform(data['y_' + category][...,
                                                                          0])
    data['train_loader'] = DataLoader(data['x_train'],
                                      data['y_train'],
                                      batch_size,
                                      shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'],
                                    data['y_val'],
                                    test_batch_size,
                                    shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'],
                                     data['y_test'],
                                     test_batch_size,
                                     shuffle=False)
    data['scaler'] = scaler

    return data


def load_dataset_hz(dataset_dir,
                    batch_size,
                    val_batch_size,
                    test_batch_size,
                    scaler_axis=(0, 1, 2, 3), **kwargs):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = load_pickle(os.path.join(dataset_dir, category + '.pkl'))
        data['x_' + category] = cat_data['x']
        data['xtime_' + category] = cat_data['xtime']
        data['y_' + category] = cat_data['y']
        data['ytime_' + category] = cat_data['ytime']

        # cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        # data['x_' + category] = cat_data['x'][:, :, :, 0:1]
        # data['xtime_' + category] = np.squeeze(np.squeeze(cat_data['x'][:, :, 0:1, 1:], axis=-2), axis=-1)
        # data['y_' + category] = cat_data['y'][:, :, :, 0:1]
        # data['ytime_' + category] = np.squeeze(np.squeeze(cat_data['y'][:, :, 0:1, 1:], axis=-2), axis=-1)

    scaler = StandardScaler(mean=data['x_train'].mean(axis=scaler_axis),
                            std=data['x_train'].std(axis=scaler_axis))
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        data['y_' + category] = scaler.transform(data['y_' + category])
    data['train_loader'] = DataLoader(data['x_train'],
                                      data['y_train'],
                                      data['xtime_train'],
                                      data['ytime_train'],
                                      batch_size,
                                      shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'],
                                    data['y_val'],
                                    data['xtime_val'],
                                    data['ytime_val'],
                                    val_batch_size,
                                    shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'],
                                     data['y_test'],
                                     data['xtime_test'],
                                     data['ytime_test'],
                                     test_batch_size,
                                     shuffle=False)
    # data['test_loader'] = DataLoader(np.concatenate([data['x_val'], data['x_test']], axis=0),
    #                                  np.concatenate([data['y_val'], data['y_test']], axis=0),
    #                                  np.concatenate([data['xtime_val'], data['xtime_test']], axis=0),
    #                                  np.concatenate([data['ytime_val'], data['ytime_test']], axis=0),
    #                                  test_batch_size,
    #                                  shuffle=False)
    data['scaler'] = scaler

    return data


class SimpleBatch(list):

    def to(self, device):
        for ele in self:
            ele.to(device)
        return self


def collate_wrapper(x, y, edge_index, edge_attr, device, xtime, ytime, return_y=True):
    x = torch.tensor(x, dtype=torch.float, device=device)
    y = torch.tensor(y, dtype=torch.float, device=device)
    x = x.transpose(dim0=1, dim1=0)  # (T, N, num_nodes, num_features)
    y_T_first = y.transpose(dim0=1, dim1=0)  # (T, N, num_nodes, num_features)
    #  do not transpose y_truth
    # T = 4, N = 8
    T = x.size()[0]
    N = x.size()[1]

    # generate batched sequence.
    sequences = []
    for t in range(T):
        cur_batch_x = x[t]
        cur_batch_y = y_T_first[t]
        batch = Batch.from_data_list([
            Data(x=cur_batch_x[i],
                 edge_index=edge_index,
                 edge_attr=edge_attr,
                 y=cur_batch_y[i],
                 xtime=xtime[i][t],
                 ytime=ytime[i][t]) for i in range(N)
        ])
        sequences.append(batch)
    if return_y:
        return SimpleBatch(sequences), y
    else:
        return SimpleBatch(sequences)


def collate_wrapper_dynamic(x, y, edge_index, edge_attr, device, xtime, ytime, return_y=True):
    x = torch.tensor(x, dtype=torch.float, device=device)
    y = torch.tensor(y, dtype=torch.float, device=device)
    x = x.transpose(dim0=1, dim1=0)  # (T, N, num_nodes, num_features)
    y_T_first = y.transpose(dim0=1, dim1=0)  # (T, N, num_nodes, num_features)
    #  do not transpose y_truth
    # T = 4, N = 8
    T = x.size()[0]
    N = x.size()[1]

    # generate batched sequence.
    sequences = []
    for t in range(T):
        cur_batch_x = x[t]
        cur_batch_y = y_T_first[t]
        batch = Batch.from_data_list([
            Data(x=cur_batch_x[i],
                 edge_index=edge_index[i],
                 edge_attr=edge_attr[i],
                 y=cur_batch_y[i],
                 xtime=xtime[i][t],
                 ytime=ytime[i][t]) for i in range(N)
        ])
        sequences.append(batch)
    if return_y:
        return SimpleBatch(sequences), y
    else:
        return SimpleBatch(sequences)


def collate_wrapper_multi_branches(x_numpy, y_numpy, edge_index_list, device):
    sequences_multi_branches = []
    for edge_index in edge_index_list:
        sequences, y = collate_wrapper(x_numpy, y_numpy, edge_index, device, return_y=True)
        sequences_multi_branches.append(sequences)

    return sequences_multi_branches, y


def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]


def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()
