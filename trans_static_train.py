import argparse
import os
import random
import time

import numpy as np
import torch
import yaml
from torch import nn
from torch import optim
from torch.nn.init import xavier_uniform_
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import MultiStepLR

from ggnn.multigraph import Net
from lib import metrics
from lib import utils
from lib.utils import collate_wrapper
from trans_utils import read_cfg_file, get_log_dir, get_logger, get_device, load_graph_data_adj_mx, load_graph_data

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# python ggnn_evaluation_hz_top20_peak.py --config data/model/trans_hz_multi_graph.yaml
# python ggnn_evaluation_sh_top72_peak.py --config data/model/trans_sh_multi_graph.yaml


def run_model(model, data_iterator, edge_index, edge_attr, device, output_dim, scaler):
    # while evaluation, we need model.eval and torch.no_grad
    model.eval()
    y_pred_list = []
    for _, (x, y, xtime, ytime) in enumerate(data_iterator):
        y = y[..., :output_dim]
        sequences, y = collate_wrapper(x=x, y=y,
                                       edge_index=edge_index,
                                       edge_attr=edge_attr,
                                       device=device,
                                       xtime=xtime,
                                       ytime=ytime)
        with torch.no_grad():
            y_pred = model(sequences, scaler)
            y_pred_list.append(y_pred.cpu().numpy())
    return y_pred_list


def evaluate(model,
             dataset,
             dataset_type,
             edge_index,
             edge_attr,
             device,
             output_dim,
             logger,
             detail=True,
             cfg=None,
             format_result=False):
    if detail:
        logger.info('Evaluation_{}_Begin:'.format(dataset_type))
    scaler = dataset['scaler']
    y_preds = run_model(
        model,
        data_iterator=dataset['{}_loader'.format(dataset_type)].get_iterator(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        device=device,
        output_dim=output_dim,
        scaler=scaler)

    y_preds = np.concatenate(y_preds, axis=0)  # concat in batch_size dim.
    mae_list = []
    mape_list = []
    rmse_list = []
    mae_sum = 0
    mape_sum = 0
    rmse_sum = 0
    horizon = cfg['model']['horizon']
    for horizon_i in range(horizon):
        y_truth = scaler.inverse_transform(
            dataset['y_{}'.format(dataset_type)][:, horizon_i, :, :output_dim])

        y_pred = scaler.inverse_transform(
            y_preds[:y_truth.shape[0], horizon_i, :, :output_dim])
        mae = metrics.masked_mae_np(y_pred, y_truth, null_val=0, mode='dcrnn')
        mape = metrics.masked_mape_np(y_pred, y_truth, null_val=0)
        rmse = metrics.masked_rmse_np(y_pred, y_truth, null_val=0)
        mae_sum += mae
        mape_sum += mape
        rmse_sum += rmse
        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)
        msg = "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}"
        if detail:
            logger.info(msg.format(horizon_i + 1, mae, mape, rmse))
    if detail:
        logger.info('Evaluation_{}_End:'.format(dataset_type))
    if format_result:
        for i in range(len(mape_list)):
            print('{:.2f}'.format(mae_list[i]))
            print('{:.2f}%'.format(mape_list[i] * 100))
            print('{:.2f}'.format(rmse_list[i]))
            print()
    else:
        return mae_sum / horizon, mape_sum / horizon, rmse_sum / horizon


class StepLR2(MultiStepLR):

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 last_epoch=-1,
                 min_lr=2.0e-6):

        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.min_lr = min_lr
        super(StepLR2, self).__init__(optimizer, milestones, gamma)

    def get_lr(self):
        lr_candidate = super(StepLR2, self).get_lr()
        if isinstance(lr_candidate, list):
            for i in range(len(lr_candidate)):
                lr_candidate[i] = max(self.min_lr, lr_candidate[i])

        else:
            lr_candidate = max(self.min_lr, lr_candidate)

        return lr_candidate


def main(args):
    cfg = read_cfg_file(args.config_filename)
    log_dir = get_log_dir(cfg)
    log_level = cfg.get('log_level', 'INFO')
    logger = get_logger(log_dir, __name__, 'info.log', level=log_level)
    device = get_device()

    adj_mx_list = []
    graph_pkl_filename = cfg['data']['graph_pkl_filename']

    for g in graph_pkl_filename:
        adj_mx = load_graph_data_adj_mx(g)

        for i in range(len(adj_mx)):
            adj_mx[i, i] = 0
        adj_mx_list.append(adj_mx)

    adj_mx = np.stack(adj_mx_list, axis=-1)
    if cfg['model'].get('norm', False):
        adj_mx = adj_mx / (adj_mx.sum(axis=0) + 1e-18)

    src, dst = adj_mx.sum(axis=-1).nonzero()
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_attr = torch.tensor(adj_mx[adj_mx.sum(axis=-1) != 0],
                             dtype=torch.float,
                             device=device)

    output_dim = cfg['model']['output_dim']

    dataset = utils.load_dataset_hz(**cfg['data'],
                                    scaler_axis=(0,
                                                 1,
                                                 2,
                                                 3))

    scaler = dataset['scaler']
    scaler_torch = utils.StandardScalerTorch(scaler.mean,
                                             scaler.std,
                                             device=device)

    model = Net(cfg).to(device)
    criterion = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg['train']['base_lr'],
                           eps=cfg['train']['epsilon'])
    scheduler = StepLR2(optimizer=optimizer,
                        milestones=cfg['train']['steps'],
                        gamma=cfg['train']['lr_decay_ratio'],
                        min_lr=cfg['train']['min_learning_rate'])

    max_grad_norm = cfg['train']['max_grad_norm']
    train_patience = cfg['train']['patience']
    val_steady_count = 0
    last_val_mae = 1e6
    horizon = cfg['model']['horizon']

    for epoch in range(cfg['train']['epochs']):
        total_loss = 0
        i = 0
        begin_time = time.perf_counter()
        train_iterator = dataset['train_loader'].get_iterator()
        model.train()
        for _, (x, y, xtime, ytime) in enumerate(train_iterator):
            y = y[:, :horizon, :, :output_dim]
            sequences, y = collate_wrapper(x=x, y=y,
                                           edge_index=edge_index,
                                           edge_attr=edge_attr,
                                           device=device,
                                           xtime=xtime,
                                           ytime=ytime)

            y_pred = model(sequences, scaler)
            y_pred = scaler_torch.inverse_transform(y_pred)
            y = scaler_torch.inverse_transform(y)
            loss = criterion(y_pred, y)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += loss.item()
            i += 1

        val_result = evaluate(model=model,
                              dataset=dataset,
                              dataset_type='val',
                              edge_index=edge_index,
                              edge_attr=edge_attr,
                              device=device,
                              output_dim=output_dim,
                              logger=logger,
                              detail=False,
                              cfg=cfg)
        val_mae, _, _ = val_result
        time_elapsed = time.perf_counter() - begin_time

        logger.info('Epoch:{}, train_mae:{:.2f}, val_mae:{}, lr={},  time_elapsed:{}'.format(
            epoch,
            total_loss / i,
            val_mae,
            str(scheduler.get_lr()),
            time_elapsed))
        if last_val_mae > val_mae:
            logger.info('val_mae decreased from {:.2f} to {:.2f}'.format(
                last_val_mae,
                val_mae))
            last_val_mae = val_mae
            val_steady_count = 0
        else:
            val_steady_count += 1

        if (epoch + 1) % cfg['train']['test_every_n_epochs'] == 0:
            evaluate(model=model,
                     dataset=dataset,
                     dataset_type='test',
                     edge_index=edge_index,
                     edge_attr=edge_attr,
                     device=device,
                     output_dim=output_dim,
                     logger=logger,
                     cfg=cfg)

        if (epoch + 1) % cfg['train']['save_every_n_epochs'] == 0:
            save_dir = log_dir
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            config_path = os.path.join(save_dir,
                                       'config-{}.yaml'.format(epoch + 1))
            epoch_path = os.path.join(save_dir,
                                      'epoch-{}.pt'.format(epoch + 1))
            torch.save(model.state_dict(), epoch_path)
            with open(config_path, 'w') as f:
                from copy import deepcopy
                save_cfg = deepcopy(cfg)
                save_cfg['model']['save_path'] = epoch_path
                f.write(yaml.dump(save_cfg, Dumper=Dumper))

        if train_patience <= val_steady_count:
            logger.info('early stopping.')
            break
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename',
                        default=None,
                        type=str,
                        help='Configuration filename for restoring the model.')
    args = parser.parse_args()
    main(args)
