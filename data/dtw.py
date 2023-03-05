import math

import numpy as np
from dtw import dtw
from numpy.linalg import norm
import pickle
import torch

mode = 'shanghai'
if mode == 'hangzhou':
    every_day = 66
    node_num = 80
    top_k = 35
else:
    every_day = 66
    node_num = 288
    top_k = 35


def get_dtw(series1, series2):
    sum_s1 = sum(series1)
    sum_s2 = sum(series2)
    series1 -= (sum_s1 - sum_s2) / len(series1)
    dist, cost, acc, path = dtw(series1, series2, dist=lambda x, y: norm(x - y, ord=1))
    return dist


def get_top_k(graph):
    graph = torch.tensor(graph)
    graph_top_k, _ = graph.topk(k=top_k, dim=1)
    graph_min = torch.min(graph_top_k, dim=-1).values
    graph_min = graph_min.unsqueeze(-1).repeat(1, node_num)
    ge = torch.ge(graph, graph_min)
    zero = torch.zeros_like(graph)
    graph = torch.where(ge, graph, zero)

    return graph.numpy()


@DeprecationWarning
def norm_graph(graph):
    graph = graph.permute(1, 0) / torch.sum(graph, dim=-1)
    graph = graph.permute(1, 0)

    return graph.numpy()


input_f = list()
output_f = list()

input_res = list()
output_res = list()

for category in ['train', 'val', 'test']:

    with open(mode + '/' + category + '.pkl', 'rb') as f:
        pickle_data = pickle.load(f)

    all_days = len(pickle_data['x']) // every_day
    for day in range(all_days):
        print(day)
        one_data = pickle_data['x'][day * every_day:(day + 1) * every_day]

        series = list()

        for i in range(0, every_day, 4):
            for j in range(4):
                series.append(one_data[i, j, :, :])
        series.append(one_data[65, 0, :, :])

        input_s = [[] for i in range(node_num)]
        output_s = [[] for j in range(node_num)]
        for o in series:
            for n in range(node_num):
                input_s[n].append([o[n, 0]])
                output_s[n].append([o[n, 1]])

        for i in range(node_num):
            input_f.append(np.array(input_s[i]))
            output_f.append(np.array(output_s[i]))

        for k in range(4, 70):
            input_graph = [[0] * node_num for _ in range(node_num)]
            output_graph = [[0] * node_num for _ in range(node_num)]
            for i in range(node_num):
                for j in range(i + 1, node_num):
                    step = math.floor(math.log(k, 4) * 4)
                    dtw_i = math.exp(-get_dtw(input_f[i][k - step:k], input_f[j][k - step:k]))
                    input_graph[i][j] = dtw_i
                    input_graph[j][i] = dtw_i

                    dtw_o = math.exp(-get_dtw(output_f[i][k - step:k], output_f[j][k - step:k]))
                    output_graph[i][j] = dtw_o
                    output_graph[j][i] = dtw_o

            input_res.append(get_top_k(input_graph))
            output_res.append(get_top_k(output_graph))

    output = open(mode + "/dynamic_sim_{0}_{1}.pkl".format(top_k, category), "wb")

    out_dic = dict()
    out_dic['input'] = input_res
    out_dic['output'] = output_res
    pickle.dump(out_dic, output)
    output.close()
