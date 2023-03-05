import collections
import os

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

station_num = 80


def get_matrix(file_name):
    data = pd.read_csv(file_name)
    # time,lineID,stationID,deviceID,status,userID,payType
    a_dict = collections.defaultdict(int)

    index_num = file_name.split('-')[2].split('.')[0]
    print(index_num)

    result = np.array([[0] * station_num for _ in range(station_num)])

    for index, row in tqdm(data.iterrows()):
        user_id = str(row['userID'])
        station_id = int(row['stationID'])

        # 因stationID为54的数据缺失，评测时stationID为54的项会被忽略，均不纳入评测范围。
        if station_id == 54:
            continue

        real_station_id = station_id if station_id < 54 else station_id - 1
        if user_id in a_dict:
            result[a_dict[user_id]][real_station_id] += 1
            del a_dict[user_id]
        else:
            a_dict[user_id] = real_station_id

    for i in range(station_num):
        result[i][i] = 0

    no_zero = 0
    for i in range(station_num):
        for j in range(station_num):
            if result[i][j] < 300:
                result[i][j] = 0
            else:
                no_zero += 1
    print(no_zero)

    row_sum = result.sum(axis=-1)
    res = list()
    for i in range(station_num):
        temp = result[i] / (row_sum[i] + 1e-18)
        res.append(temp.tolist())

    output = open("hz_correlation_{0}.pkl".format(index_num), "wb")
    pickle.dump(res, output)
    output.close()


def get_all_pkl():
    res = list()
    for filepath, _, filenames in os.walk(r'D:\traffic-forecost-code\TransGCN\data_deal'):
        for filename in filenames:
            if filename.endswith('pkl'):
                res.append(filepath + '\\' + filename)

    return res


for filepath, _, filenames in os.walk(r'F:\天池城市AI地铁客流量预测\Metro_train'):
    for filename in filenames:
        get_matrix(filepath + '\\' + filename)

all_dir = get_all_pkl()

all_data = list()
for dir in all_dir:
    df = open(dir, 'rb')
    dd = pickle.load(df)
    all_data.append(dd)
    df.close()

output = open("hz_correlation_{0}.pkl".format('all'), "wb")
pickle.dump(all_data, output)
output.close()
