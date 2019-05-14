# encoding: utf-8

from jaqs.data import RemoteDataService, DataView
import tushare as ts
from jaqs.data import DataApi
import numpy as np
import pandas as pd
import talib
import os
import multiprocessing
import matplotlib.pyplot as plt
from random import sample
from shutil import copyfile

import trendline
import draw_indictors

import time

data_config = {
  "remote.data.address": "tcp://data.quantos.org:8910",
  "remote.data.username": "18652420434",
  "remote.data.password": "eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MTcwNjAxMDgyOTMiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg2NTI0MjA0MzQifQ.b1ejSpbEVS7LhbsveZ5kvbWgUs7fnUd0-CBakPwNUu4"
}
trade_config = {
  "remote.trade.address": "tcp://gw.quantos.org:8901",
  "remote.trade.username": "18652420434",
  "remote.trade.password": "eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MTcwNjAxMDgyOTMiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg2NTI0MjA0MzQifQ.b1ejSpbEVS7LhbsveZ5kvbWgUs7fnUd0-CBakPwNUu4"
}

dataview_store_folder = './read_picture/data/prepared'

api = DataApi(addr='tcp://data.quantos.org:8910')
api.login(18652420434, 'eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MTcwNjAxMDgyOTMiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg2NTI0MjA0MzQifQ.b1ejSpbEVS7LhbsveZ5kvbWgUs7fnUd0-CBakPwNUu4')

def get_index_info():
    df, msg = api.query(
                view="lb.indexCons", 
                fields="index_code,symbol,in_date,out_date", 
                filter="index_code=000300.SH&start_date=20120101&end_date=20181231", 
                data_format='pandas')
    print("get_index_info: " +msg)
    return df

def get_sec_susp():
    df, msg = api.query(
                view="lb.secSusp", 
                fields="susp_date,resu_date,susp_reason,resu_time", 
                filter="symbol=600410.SH&start_date=20080101&end_date=20181231", 
                data_format='pandas')
    print(msg)
    # df.to_csv(path_or_buf='./read_picture/data/csv/stop.csv', sep=',', index=True)
    return df

def download_data():
    dataview_props = {'start_date': 20120101, 'end_date': 20181231,
                      'universe': '000300.SH',
                    #   'symbol':'600030.SH,600104.SH',
                      'fields': 'open,close,high,low,close_adj,volume',
                      'freq': 1}

    ds = RemoteDataService()
    ds.init_from_config(data_config)

    # DataView utilizes RemoteDataService to get various data and store them
    dv = DataView()
    dv.init_from_config(dataview_props, ds)
    dv.prepare_data()

    factor_formula = 'Delay(Return(close_adj, 2, 0), -2)'
    dv.add_formula('future_return_2', factor_formula, is_quarterly=False, is_factor=False)
    factor_formula = 'Delay(Return(close_adj, 3, 0), -3)'
    dv.add_formula('future_return_3', factor_formula, is_quarterly=False, is_factor=False)
    factor_formula = 'Delay(Return(close_adj, 4, 0), -4)'
    dv.add_formula('future_return_4', factor_formula, is_quarterly=False, is_factor=False)
    factor_formula = 'Delay(Return(close_adj, 5, 0), -5)'
    dv.add_formula('future_return_5', factor_formula, is_quarterly=False, is_factor=False)

    dv.save_dataview(folder_path=dataview_store_folder)


def save_data_to_csv():
    dv = DataView()
    dv.load_dataview(folder_path=dataview_store_folder)

    # df = pd.DataFrame()

    # df['close'] = dv.get_ts('close', symbol=symbol, start_date=20080101, end_date=20171231)[symbol]
    # df['open']  = dv.get_ts('open', symbol=symbol, start_date=20080101, end_date=20171231)[symbol]
    # df['high']  = dv.get_ts('high', symbol=symbol, start_date=20080101, end_date=20171231)[symbol]
    # df['low']   = dv.get_ts('low', symbol=symbol, start_date=20080101, end_date=20171231)[symbol]

    # df = df.dropna()
    # snap1 = dv.get_snapshot(20080424, symbol='600030.SH', fields='open,close,high,low,volume')
    # ts1 = dv.get_ts('open,close,high,low,close_adj,future_return_2,future_return_3,future_return_4,future_return_5', symbol='600030.SH', start_date=20080101, end_date=20080302)
    sh_000905 = get_index_info()
    for symbol in sh_000905['symbol']:
    # for symbol in ['600030.SH', '600104.SH']:
        print(symbol)
        ts_symbol = dv.get_ts('open,close,high,low,volume,future_return_2,future_return_3,future_return_4,future_return_5', symbol=symbol, start_date=start_date, end_date=end_time)[symbol]

        ts_symbol.fillna(0, replace=True)
        ts_symbol = ts_symbol[(ts_symbol[['volume']] != 0).all(axis=1)]

        ts_symbol['date'] = ts_symbol.index
        ts_symbol['date'] = pd.to_datetime(ts_symbol['date'], format='%Y%m%d')
        ts_symbol = ts_symbol.reset_index(drop=True)
        _kdj = trendline.kdj(ts_symbol)
        _macd = trendline.macd(ts_symbol)
        _rsi = trendline.rsi(ts_symbol)
        _vrsi = trendline.vrsi(ts_symbol)
        _boll = trendline.boll(ts_symbol)
        _bbiboll = trendline.bbiboll(ts_symbol)
        _wr = trendline.wr(ts_symbol)
        _bias = trendline.bias(ts_symbol)
        _asi = trendline.asi(ts_symbol)
        _vr_rate = trendline.vr_rate(ts_symbol)
        _vr = trendline.vr(ts_symbol)
        _arbr = trendline.arbr(ts_symbol)
        _dpo = trendline.dpo(ts_symbol)
        _trix = trendline.trix(ts_symbol)
        _bbi = trendline.bbi(ts_symbol)
        _mtm = trendline.mtm(ts_symbol)
        _obv = trendline.obv(ts_symbol)
        _cci = trendline.cci(ts_symbol)
        _priceosc = trendline.priceosc(ts_symbol)
        _dbcd = trendline.dbcd(ts_symbol)
        _roc = trendline.roc(ts_symbol)
        _vroc = trendline.vroc(ts_symbol)
        _cr = trendline.cr(ts_symbol)
        _psy = trendline.psy(ts_symbol)
        _wad = trendline.wad(ts_symbol)
        _mfi = trendline.mfi(ts_symbol)
        _vosc = trendline.vosc(ts_symbol)
        # _jdqs = trendline.jdqs(ts_symbol)
        # _jdrs = trendline.jdrs(ts_symbol)

        ts_symbol = trendline.join_frame(ts_symbol,_kdj)
        ts_symbol = trendline.join_frame(ts_symbol, _macd)
        ts_symbol = trendline.join_frame(ts_symbol, _rsi)
        ts_symbol = trendline.join_frame(ts_symbol, _vrsi)
        ts_symbol = trendline.join_frame(ts_symbol, _boll)
        ts_symbol = trendline.join_frame(ts_symbol, _bbiboll)
        ts_symbol = trendline.join_frame(ts_symbol, _wr)
        ts_symbol = trendline.join_frame(ts_symbol, _bias)
        ts_symbol = trendline.join_frame(ts_symbol, _asi)
        ts_symbol = trendline.join_frame(ts_symbol, _vr_rate)
        ts_symbol = trendline.join_frame(ts_symbol, _vr)
        ts_symbol = trendline.join_frame(ts_symbol, _arbr)
        ts_symbol = trendline.join_frame(ts_symbol, _dpo)
        ts_symbol = trendline.join_frame(ts_symbol, _trix)
        ts_symbol = trendline.join_frame(ts_symbol, _bbi)
        ts_symbol = trendline.join_frame(ts_symbol, _mtm)
        ts_symbol = trendline.join_frame(ts_symbol, _obv)
        ts_symbol = trendline.join_frame(ts_symbol, _cci)
        ts_symbol = trendline.join_frame(ts_symbol, _priceosc)
        ts_symbol = trendline.join_frame(ts_symbol, _dbcd)
        ts_symbol = trendline.join_frame(ts_symbol, _roc)
        ts_symbol = trendline.join_frame(ts_symbol, _vroc)
        ts_symbol = trendline.join_frame(ts_symbol, _cr)
        ts_symbol = trendline.join_frame(ts_symbol, _psy)
        ts_symbol = trendline.join_frame(ts_symbol, _wad)
        ts_symbol = trendline.join_frame(ts_symbol, _mfi)
        ts_symbol = trendline.join_frame(ts_symbol, _vosc)
        # ts_symbol = trendline.join_frame(ts_symbol, _jdqs)
        # ts_symbol = trendline.join_frame(ts_symbol, _jdrs)

        save_csv(symbol, ts_symbol)

    # ts1 = dv.get_ts('open,close,high,low,volume', symbol='600030.SH', start_date=20080101, end_date=20081002)['600030.SH']
    # ts1['date'] = ts1.index
    # ts1['date'] = pd.to_datetime(ts1['date'], format='%Y%m%d')
    # ts1 = ts1.reset_index(drop=True)
    # print(ts1)
    # trendline.plot_kdj(ts1)
    # ts1.to_csv(path_or_buf='./read_picture/data/csv/600030.SH.csv', sep=',', index=True)
    # return df


def save_csv(symbol, df):
    df.to_csv(path_or_buf='./read_picture/data/csv/' + symbol + '.csv', sep=',', index=True)


def choice_pics(class_path):
    class_path_up = class_path + 'up/'
    class_path_down = class_path + 'down/'
    class_path_equal = class_path + 'equal/'
    file_count = 0
    file_list_up = os.listdir(class_path_up)
    file_list_down = os.listdir(class_path_down)
    file_list_equal = os.listdir(class_path_equal)
    if len(file_list_down) > len(file_list_up):
        file_count = len(file_list_up)
    else:
        file_count = len(file_list_down)
    print(file_count)
    file_train_count = int(file_count*0.7)
    print(file_train_count)

    # 从目录中取一样多个图片
    file_list_up = sample(file_list_up,file_count)
    file_list_down = sample(file_list_down,file_count)
    file_list_equal = sample(file_list_equal,file_count)
    # 训练图片与测试图片7,3分
    file_list_up_train = sample(file_list_up,file_train_count)
    file_list_up_val = list(set(file_list_up) - set(file_list_up_train))
    file_list_down_train = sample(file_list_down,file_train_count)
    file_list_down_val = list(set(file_list_down) - set(file_list_down_train))
    file_list_equal_train = sample(file_list_equal,file_train_count)
    file_list_equal_val = list(set(file_list_equal) - set(file_list_equal_train))

    for i, img_name in enumerate(file_list_up_train):
        img_path = class_path_up + img_name
        copyfile(img_path, './data/train/up/' + img_name)
    for i, img_name in enumerate(file_list_down_train):
        img_path = class_path_down + img_name
        copyfile(img_path, './data/train/down/' + img_name)
    for i, img_name in enumerate(file_list_equal_train):
        img_path = class_path_equal + img_name
        copyfile(img_path, './data/train/equal/' + img_name)

    for i, img_name in enumerate(file_list_up_val):
        img_path = class_path_up + img_name
        copyfile(img_path, './data/validation/up/' + img_name)
    for i, img_name in enumerate(file_list_down_val):
        img_path = class_path_down + img_name
        copyfile(img_path, './data/validation/down/' + img_name)
    for i, img_name in enumerate(file_list_equal_val):
        img_path = class_path_equal + img_name
        copyfile(img_path, './data/validation/equal/' + img_name)


# # get_data()
# # draw_pic('600703.SH', 'macd_j')
# # choice_pics('./data/prepared/pic_data/macd_j_pic2/')
# # download_data()
# # save_data_to_csv()

# df = pd.read_csv('./read_picture/data/csv/600267.SH.csv', sep=',')
# df = df[(df[['volume']] != 0).all(axis=1)]
# print(len(df))
# print(df[2608: 2618])
# df = pd.read_csv('./read_picture/data/csv/000012.SZ.csv', sep=',')
# df = df[(df[['volume']] != 0).all(axis=1)]
# df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
# # df = df[df['date'] > pd.datetime(2018, 10, 31)]

# ss = time.time()
# for i in range(20):
#     start = time.time()
#     draw_indictors.plot_all(df[35+i: 45+i], is_show=False, output='./read_picture/data/img/xxx.png')
#     print("everytime: " + str(time.time() - start))
# print("total: " + str(time.time() - ss))



# for i in range(len(df) - 7):
#     output = './read_picture/data/img/' + str(i) + '.png'
#     # print(df[35+i:36+i]['future_return_4'])
#     draw_indictors.plot_all(df[35+i: 45+i], is_show=False, output=output)




# if __name__ == '__main__':
#     df = pd.read_csv('./read_picture/data/csv/000012.SZ.csv', sep=',')
#     df = df[(df[['volume']] != 0).all(axis=1)]
#     df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
#     # df = df[df['date'] > pd.datetime(2018, 10, 31)]

#     ss = time.time()
#     for i in range(20):
#         start = time.time()
#         draw_indictors.plot_all(df[35+i: 45+i], is_show=False, output='./read_picture/data/img/xxx.png')
#         print("everytime: " + str(time.time() - start))
#     print("total: " + str(time.time() - ss))


#     multiprocessing.freeze_support()
#     pool = multiprocessing.Pool()
#     ss = time.time()
#     for i in range(20):
#         start = time.time()
#         pool.apply(draw_indictors.plot_all, args=(df[35+i: 45+i], False, './read_picture/data/img/xxx.png'))
#         print("everytime: " + str(time.time() - start))
#     print("total: " + str(time.time() - ss))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()

    download_data()
    save_data_to_csv()

    count = 0

    # sh_000905 = {'symbol':['000950.SZ']}
    sh_000300 = get_index_info()
    # print(sh_000905)
    for symbol in sh_000300['symbol']:
        start = time.time()
        print(symbol)
        count += 1
        print(str(count)+"/500")
        df = pd.read_csv('./read_picture/data/csv/'+symbol+'.csv', sep=',')

        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

        # df.to_csv('./read_picture/data/csv/'+symbol+'_.csv', sep=',')

        if not os.path.exists('./read_picture/data/img/'+symbol+'/up'):
            os.makedirs('./read_picture/data/img/'+symbol+'/up')
        if not os.path.exists('./read_picture/data/img/'+symbol+'/down'):
            os.makedirs('./read_picture/data/img/'+symbol+'/down')
        
        count = len(df) - 10 - 35
        for i in range(500):
            ts = df[35+i: 45+i]
            if(ts.head(1)['future_return_4'].values[0] > 0):
                output = './read_picture/data/img/'+symbol+'/up/' + str(i) + '.png'
            elif(ts.head(1)['future_return_4'].values[0] < 0):
                output = './read_picture/data/img/'+symbol+'/down/' + str(i) + '.png'
            else:
                continue
            pool.apply_async(draw_indictors.plot_all, args=(ts, False, output))
            # print(str(i) + "/" + str(count))
            # draw_indictors.plot_all(df[35+i: 45+i], is_show=False, output=output)
        pool.close()
        pool.join()
        end = time.time()
        print("耗时：" + str(end - start))
        
