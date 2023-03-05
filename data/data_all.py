import os
from trans_utils import load_pickle
import math
import numpy as np
from dateutil.parser import parse
import datetime

all_data_dic = dict()
mode = 'hangzhou'
# mode = 'shanghai'

date_hz_list = ['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04', '2019-01-05', '2019-01-06', '2019-01-07',
                '2019-01-08', '2019-01-09', '2019-01-10', '2019-01-11', '2019-01-12', '2019-01-13', '2019-01-14',
                '2019-01-15', '2019-01-16', '2019-01-17', '2019-01-18', '2019-01-19', '2019-01-20', '2019-01-21',
                '2019-01-22', '2019-01-23', '2019-01-24', '2019-01-25']

date_sh_list = ['2016-07-01', '2016-07-02', '2016-07-03', '2016-07-04', '2016-07-05', '2016-07-06', '2016-07-07',
                '2016-07-08', '2016-07-09', '2016-07-10', '2016-07-11', '2016-07-12', '2016-07-13', '2016-07-14',
                '2016-07-15', '2016-07-16', '2016-07-17', '2016-07-18', '2016-07-19', '2016-07-20', '2016-07-21',
                '2016-07-22', '2016-07-23', '2016-07-24', '2016-07-25', '2016-07-26', '2016-07-27', '2016-07-28',
                '2016-07-29', '2016-07-30', '2016-07-31',
                '2016-08-01', '2016-08-02', '2016-08-03', '2016-08-04', '2016-08-05', '2016-08-06', '2016-08-07',
                '2016-08-08', '2016-08-09', '2016-08-10', '2016-08-11', '2016-08-12', '2016-08-13', '2016-08-14',
                '2016-08-15', '2016-08-16', '2016-08-17', '2016-08-18', '2016-08-19', '2016-08-20', '2016-08-21',
                '2016-08-22', '2016-08-23', '2016-08-24', '2016-08-25', '2016-08-26', '2016-08-27', '2016-08-28',
                '2016-08-29', '2016-08-30', '2016-08-31',
                '2016-09-01', '2016-09-02', '2016-09-03', '2016-09-04', '2016-09-05', '2016-09-06', '2016-09-07',
                '2016-09-08', '2016-09-09', '2016-09-10', '2016-09-11', '2016-09-12', '2016-09-13', '2016-09-14',
                '2016-09-15', '2016-09-16', '2016-09-17', '2016-09-18', '2016-09-19', '2016-09-20', '2016-09-21',
                '2016-09-22', '2016-09-23', '2016-09-24', '2016-09-25', '2016-09-26', '2016-09-27', '2016-09-28',
                '2016-09-29', '2016-09-30']
num = 1
date_list = date_hz_list
if mode == 'shanghai':
    date_list = date_sh_list

for i in date_list:
    all_data_dic[i] = dict()

for category in ['train', 'val', 'test']:
    cat_data = load_pickle(os.path.join('data/' + mode, category + '.pkl'))
    for i, item in enumerate(cat_data['y']):
        all_data_dic[date_list[int(math.ceil(num / 66)) - 1]][
            str(cat_data['ytime'][i:i + 1, :1][0][0]).split('T')[1]] = item
        num += 1

gt = parse("2016-07-08")
lt = parse("2016-09-09")


def get_all_date(time_info):
    ymd = str(np.datetime64(time_info)).split('T')[0]
    hms = str(np.datetime64(time_info)).split('T')[1]

    if mode == 'hangzhou':
        if '2019-01-01' == ymd:
            return (all_data_dic['2019-01-06'][hms] + all_data_dic['2019-01-13'][hms]) / 2
        elif '2019-01-02' == ymd:
            return (all_data_dic['2019-01-03'][hms] + all_data_dic['2019-01-09'][hms]) / 2
        elif '2019-01-03' == ymd:
            return (all_data_dic['2019-01-02'][hms] + all_data_dic['2019-01-10'][hms]) / 2
        elif '2019-01-04' == ymd:
            return (all_data_dic['2019-01-03'][hms] + all_data_dic['2019-01-11'][hms]) / 2
        elif '2019-01-05' == ymd:
            return all_data_dic['2019-01-12'][hms]
        elif '2019-01-06' == ymd:
            return all_data_dic['2019-01-13'][hms]
        elif '2019-01-07' == ymd:
            return (all_data_dic['2019-01-08'][hms] + all_data_dic['2019-01-14'][hms]) / 2
        elif '2019-01-08' == ymd:
            return (all_data_dic['2019-01-07'][hms] + all_data_dic['2019-01-15'][hms]) / 2
        elif '2019-01-09' == ymd:
            return (all_data_dic['2019-01-02'][hms] + all_data_dic['2019-01-08'][hms]) / 2
        elif '2019-01-10' == ymd:
            return (all_data_dic['2019-01-03'][hms] + all_data_dic['2019-01-09'][hms]) / 2
        elif '2019-01-11' == ymd:
            return (all_data_dic['2019-01-04'][hms] + all_data_dic['2019-01-10'][hms]) / 2
        elif '2019-01-12' == ymd:
            return all_data_dic['2019-01-05'][hms]
        elif '2019-01-13' == ymd:
            return all_data_dic['2019-01-06'][hms]
        elif '2019-01-14' == ymd:
            return (all_data_dic['2019-01-07'][hms] + all_data_dic['2019-01-15'][hms]) / 2
        elif '2019-01-15' == ymd:
            return (all_data_dic['2019-01-08'][hms] + all_data_dic['2019-01-14'][hms]) / 2
        elif '2019-01-16' == ymd:
            return (all_data_dic['2019-01-09'][hms] + all_data_dic['2019-01-15'][hms]) / 2
        elif '2019-01-17' == ymd:
            return (all_data_dic['2019-01-10'][hms] + all_data_dic['2019-01-16'][hms]) / 2
        elif '2019-01-18' == ymd:
            return (all_data_dic['2019-01-11'][hms] + all_data_dic['2019-01-17'][hms]) / 2
        elif '2019-01-19' == ymd:
            return (all_data_dic['2019-01-05'][hms] + all_data_dic['2019-01-12'][hms]) / 2
        elif '2019-01-20' == ymd:
            return (all_data_dic['2019-01-06'][hms] + all_data_dic['2019-01-13'][hms]) / 2
        elif '2019-01-21' == ymd:
            return (all_data_dic['2019-01-07'][hms] + all_data_dic['2019-01-14'][hms]) / 2
        elif '2019-01-22' == ymd:
            return (all_data_dic['2019-01-15'][hms] + all_data_dic['2019-01-21'][hms]) / 2
        elif '2019-01-23' == ymd:
            return (all_data_dic['2019-01-16'][hms] + all_data_dic['2019-01-22'][hms]) / 2
        elif '2019-01-24' == ymd:
            return (all_data_dic['2019-01-17'][hms] + all_data_dic['2019-01-23'][hms]) / 2
        elif '2019-01-25' == ymd:
            return (all_data_dic['2019-01-18'][hms] + all_data_dic['2019-01-24'][hms]) / 2
    else:
        if '2016-07-01' == ymd:
            return (all_data_dic['2016-07-04'][hms] + all_data_dic['2016-07-05'][hms]) / 2
        elif '2016-07-02' == ymd:
            return all_data_dic['2016-07-03'][hms]
        elif '2016-07-03' == ymd:
            return all_data_dic['2016-07-02'][hms]
        elif '2016-07-04' == ymd:
            return (all_data_dic['2016-07-05'][hms] + all_data_dic['2016-07-06'][hms]) / 2
        elif '2016-07-05' == ymd:
            return (all_data_dic['2016-07-04'][hms] + all_data_dic['2016-07-06'][hms]) / 2
        elif '2016-07-06' == ymd:
            return (all_data_dic['2016-07-04'][hms] + all_data_dic['2016-07-05'][hms]) / 2
        elif '2016-07-07' == ymd:
            return (all_data_dic['2016-07-05'][hms] + all_data_dic['2016-07-06'][hms]) / 2
        elif gt <= datetime.datetime.strptime(ymd, '%Y-%m-%d') <= lt:
            week = datetime.datetime.strptime(ymd, "%Y-%m-%d").weekday() + 1
            today = datetime.datetime.strptime(ymd, '%Y-%m-%d')
            one_day = datetime.timedelta(days=1)
            one_week = datetime.timedelta(days=7)

            if week == 1:
                return all_data_dic[(today - one_week).strftime('%Y-%m-%d')][hms]
            elif week < 6:
                return (all_data_dic[(today - one_day).strftime('%Y-%m-%d')][hms] +
                        all_data_dic[(today - one_week).strftime('%Y-%m-%d')][hms]) / 2
            elif week == 6:
                return all_data_dic[(today - one_week).strftime('%Y-%m-%d')][hms]
            elif week == 7:
                return all_data_dic[(today - one_week).strftime('%Y-%m-%d')][hms]
        if '2016-09-10' == ymd:
            return all_data_dic['2016-09-03'][hms]
        if '2016-09-11' == ymd:
            return all_data_dic['2016-09-04'][hms]
        if '2016-09-12' == ymd:
            return (all_data_dic['2016-09-05'][hms] + all_data_dic['2016-09-06'][hms]) / 2
        if '2016-09-13' == ymd:
            return (all_data_dic['2016-09-06'][hms] + all_data_dic['2016-09-12'][hms]) / 2
        if '2016-09-14' == ymd:
            return (all_data_dic['2016-09-07'][hms] + all_data_dic['2016-09-13'][hms]) / 2
        if '2016-09-15' == ymd:
            return all_data_dic['2016-09-11'][hms]
        if '2016-09-16' == ymd:
            return all_data_dic['2016-09-10'][hms]
        if '2016-09-17' == ymd:
            return all_data_dic['2016-09-11'][hms]
        if '2016-09-18' == ymd:
            return (all_data_dic['2016-09-13'][hms] + all_data_dic['2016-09-14'][hms]) / 2
        if '2016-09-19' == ymd:
            return (all_data_dic['2016-09-12'][hms] + all_data_dic['2016-09-18'][hms]) / 2
        if '2016-09-20' == ymd:
            return (all_data_dic['2016-09-13'][hms] + all_data_dic['2016-09-19'][hms]) / 2
        if '2016-09-21' == ymd:
            return (all_data_dic['2016-09-14'][hms] + all_data_dic['2016-09-20'][hms]) / 2
        if '2016-09-22' == ymd:
            return (all_data_dic['2016-09-20'][hms] + all_data_dic['2016-09-21'][hms]) / 2
        if '2016-09-23' == ymd:
            return (all_data_dic['2016-09-21'][hms] + all_data_dic['2016-09-22'][hms]) / 2
        if '2016-09-24' == ymd:
            return all_data_dic['2016-09-10'][hms]
        if '2016-09-25' == ymd:
            return all_data_dic['2016-09-11'][hms]
        if '2016-09-26' == ymd:
            return (all_data_dic['2016-09-19'][hms] + all_data_dic['2016-09-23'][hms]) / 2
        if '2016-09-27' == ymd:
            return (all_data_dic['2016-09-20'][hms] + all_data_dic['2016-09-26'][hms]) / 2
        if '2016-09-28' == ymd:
            return (all_data_dic['2016-09-21'][hms] + all_data_dic['2016-09-27'][hms]) / 2
        if '2016-09-29' == ymd:
            return (all_data_dic['2016-09-22'][hms] + all_data_dic['2016-09-28'][hms]) / 2
        if '2016-09-30' == ymd:
            return (all_data_dic['2016-09-23'][hms] + all_data_dic['2016-09-29'][hms]) / 2
