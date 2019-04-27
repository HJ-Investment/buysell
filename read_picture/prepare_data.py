# encoding: utf-8

from jaqs.data import RemoteDataService, DataView
import tushare as ts
from jaqs.data import DataApi
import numpy as np
import pandas as pd
import talib
import os
import matplotlib.pyplot as plt
from random import sample
from shutil import copyfile

import trendline
import draw_indictors

start_date = '20080101'
end_time = '20181231'

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
                filter="index_code=000905.SH&start_date=20080101&end_date=20181231", 
                data_format='pandas')
    print(msg)
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
    dataview_props = {'start_date': 20080101, 'end_date': 20181231,
                      'universe': '000905.SH',
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


def prepare_kdj(df, n, ksgn='close'):
    '''
        【输入】
            df, pd.dataframe格式数据源
            n，时间长度
            ksgn，列名，一般是：close收盘价
        【输出】
            df, pd.dataframe格式数据源,
            增加了一栏：_{n}，输出数据
    '''
    low_list = df['low'].rolling(window=n, center=False).min()  # pd.rolling_min(df['low'], n)
    low_list.fillna(value=df['low'].expanding(min_periods=1).min(), inplace=True)
    high_list = df['high'].rolling(window=n, center=False).max()  # pd.rolling_max(df['high'], n)
    high_list.fillna(value=df['high'].expanding(min_periods=1).max(), inplace=True)
    rsv = (df[ksgn] - low_list) / (high_list - low_list) * 100

    df['k'] = pd.ewma(rsv, com=2)
    df['d'] = pd.ewma(df['k'], com=2)
    df['j'] = 3.0 * df['k'] - 2.0 * df['d']
    # print('n df',len(df))
    return df


def calculator_close(df):
    df = df.reset_index()
    close_five = df['close'][5:]
    # print(close_five)
    df['close_five_value'] = close_five.reset_index(drop=True)
    # print(df)
    df['close_five'] = (df['close_five_value'] - df['close']) / df['close']
    return df


def save_csv(symbol, df):
    df.to_csv(path_or_buf='./read_picture/data/csv/' + symbol + '.csv', sep=',', index=True)


def get_data(symbol=None):
    if not os.path.exists('./data/prepared/datacsv/'):
        os.makedirs('./data/prepared/datacsv/')

    if not symbol:
        api = DataApi(addr="tcp://data.quantos.org:8910")
        result, msg = api.login("18652420434", "eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MTcwNjAxMDgyOTMiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg2NTI0MjA0MzQifQ.b1ejSpbEVS7LhbsveZ5kvbWgUs7fnUd0-CBakPwNUu4")
        # print(result)
        data, msg = api.query(
                view="lb.indexCons", 
                fields="symbol", 
                filter="index_code=000016.SH&start_date=20180801&end_date=20180831", 
                data_format='pandas')
        print(msg)
        # symbols = data['symbol'].tolist()
        symbols = ['600887.SH', '601988.SH', '600048.SH', '601006.SH', '601398.SH', '601628.SH', '601166.SH', '601318.SH', '601328.SH', '601169.SH', '601088.SH', '601857.SH', '601390.SH', '601601.SH', '601186.SH', '601668.SH', '601766.SH', '600999.SH', '601989.SH', '601688.SH', '601288.SH', '601818.SH', '601800.SH', '601360.SH', '601336.SH', '603993.SH', '601211.SH', '600958.SH', '601878.SH', '601229.SH', '601881.SH']
        print(symbols)

        for sym in symbols:
            # df = load_data(sym)
            # close = [float(x) for x in df['close']]
            # # prepare macd data
            # df['MACD'], df['MACDsignal'], df['MACDhist'] = talib.MACD(np.array(close),
            #                                                           fastperiod=12, slowperiod=26, signalperiod=9)
            # df = df.sort_index()
            # df.index = pd.to_datetime(df.index, format='%Y%m%d')
            # df = prepare_kdj(df, 9, 'close')  # 计算好kdj之后从13行开始取数据,计算出来的kdj比较准确
            # df = df[34:]
            # df = calculator_close(df)
            # save_csv(sym, df)
            draw_pic(sym, 'macd_j')

    else:
        df = load_data(symbol)
        close = [float(x) for x in df['close']]
        # prepare macd data
        df['MACD'], df['MACDsignal'], df['MACDhist'] = talib.MACD(np.array(close),
                                                                  fastperiod=12, slowperiod=26, signalperiod=9)
        # df = df.sort_index()
        # df.index = pd.to_datetime(df.index, format='%Y%m%d')
        df = prepare_kdj(df, 9, 'close')  # 计算好kdj之后从13行开始取数据,计算出来的kdj比较准确
        df = df[34:]
        df = calculator_close(df)
        save_csv(symbol, df)
        draw_pic(symbol, 'macd_j')


def draw_kdj_pic(symbol, df, sequence):
    # if not os.path.exists('./data/prepared/pic_data/kdj_pic/' + symbol):
    #     os.makedirs('./data/prepared/pic_data/kdj_pic/' + symbol)
    if not os.path.exists('./data/prepared/pic_data/kdj_pic/up'):
        os.makedirs('./data/prepared/pic_data/kdj_pic/up')
    if not os.path.exists('./data/prepared/pic_data/kdj_pic/equal'):
        os.makedirs('./data/prepared/pic_data/kdj_pic/equal')
    if not os.path.exists('./data/prepared/pic_data/kdj_pic/down'):
        os.makedirs('./data/prepared/pic_data/kdj_pic/down')
    # fig = plt.gcf()
    fig = plt.figure(figsize=(12.8, 12.8))  # 设置图形的大小,figsize=(12.8,12.8) 保存的时候dpi=10可以得到128*128的图片
    sig_k = df.k
    sig_d = df.d
    sig_j = df.k*3 - df.d*2
    plt.plot(sig_k.index, sig_k, label='k')
    plt.plot(sig_d.index, sig_d, label='d')
    plt.plot(sig_j.index, sig_j, label='j')
    plt.axis('off')
    # plt.show()
    # print(df['close_five'][4])
    str_symbol = symbol[:-3]
    if df['close_five'][2] >= 0.05:
        fig.savefig('./data/prepared/pic_data/kdj_pic/up/' + str_symbol + '_' + str(sequence), dpi=20)
    elif df['close_five'][2] <= -0.05:
        fig.savefig('./data/prepared/pic_data/kdj_pic/down/' + str_symbol + '_' + str(sequence), dpi=20)
    else:
        fig.savefig('./data/prepared/pic_data/kdj_pic/equal/' + str_symbol + '_' + str(sequence), dpi=20)
    plt.close()

    return 1


def create_macd_pic(symbol, df, sequence):
    if not os.path.exists('./data/prepared/pic_data/macd_pic/up'):
        os.makedirs('./data/prepared/pic_data/macd_pic/up')
    if not os.path.exists('./data/prepared/pic_data/macd_pic/equal'):
        os.makedirs('./data/prepared/pic_data/macd_pic/equal')
    if not os.path.exists('./data/prepared/pic_data/macd_pic/down'):
        os.makedirs('./data/prepared/pic_data/macd_pic/down')

    fig = plt.figure(figsize=(12.8,12.8))

    plt.plot(df.index, df['MACD'], label='macd dif')  # 快线
    plt.plot(df.index, df['MACDsignal'], label='signal dea')  # 慢线
    plt.bar(df.index, df['MACDhist']*2, label='hist bar')
    # plt.axis('off')

    str_symbol = symbol[:-3]
    if df['close_five'][2] >= 0.05:
        fig.savefig('./data/prepared/pic_data/macd_pic/up/' + str_symbol + '_' + str(sequence), dpi=20)
    elif df['close_five'][2] <= -0.05:
        fig.savefig('./data/prepared/pic_data/macd_pic/down/' + str_symbol + '_' + str(sequence), dpi=20)
    else:
        fig.savefig('./data/prepared/pic_data/macd_pic/equal/' + str_symbol + '_' + str(sequence), dpi=20)
    plt.close()

    return 1

def create_fixed_pic(symbol, df, sequence):
    if not os.path.exists('./data/prepared/pic_data/fix_pic/up'):
        os.makedirs('./data/prepared/pic_data/fix_pic/up')
    if not os.path.exists('./data/prepared/pic_data/fix_pic/equal'):
        os.makedirs('./data/prepared/pic_data/fix_pic/equal')
    if not os.path.exists('./data/prepared/pic_data/fix_pic/down'):
        os.makedirs('./data/prepared/pic_data/fix_pic/down')
    
    fig = plt.figure(figsize=(12.8, 12.8))  # 设置图形的大小,figsize=(12.8,12.8) 保存的时候dpi=10可以得到128*128的图片
    
    ax1 = fig.add_subplot(111)

    sig_k = df.k
    sig_d = df.d
    sig_j = df.k*3 - df.d*2
    ax1.plot(sig_k.index, sig_k, label='k')
    ax1.plot(sig_d.index, sig_d, label='d')
    ax1.plot(sig_j.index, sig_j, label='j')
    ax1.axis('off')

    ax2 = ax1.twinx()
    ax2.plot(df.index, df['MACD'], label='macd dif')  # 快线
    ax2.plot(df.index, df['MACDsignal'], label='signal dea')  # 慢线
    ax2.axis('off')

    str_symbol = symbol[:-3]
    if df['close_five'][2] >= 0.05:
        fig.savefig('./data/prepared/pic_data/fix_pic/up/' + str_symbol + '_' + str(sequence), dpi=20)
    elif df['close_five'][2] <= -0.05:
        fig.savefig('./data/prepared/pic_data/fix_pic/down/' + str_symbol + '_' + str(sequence), dpi=20)
    else:
        fig.savefig('./data/prepared/pic_data/fix_pic/equal/' + str_symbol + '_' + str(sequence), dpi=20)
    plt.close()

    return 1

def create_macd_j_pic(symbol, df, sequence):
    if not os.path.exists('./data/prepared/pic_data/macd_j_pic/up'):
        os.makedirs('./data/prepared/pic_data/macd_j_pic/up')
    if not os.path.exists('./data/prepared/pic_data/macd_j_pic/equal'):
        os.makedirs('./data/prepared/pic_data/macd_j_pic/equal')
    if not os.path.exists('./data/prepared/pic_data/macd_j_pic/down'):
        os.makedirs('./data/prepared/pic_data/macd_j_pic/down')
    fig = plt.figure(figsize=(12.8, 12.8))
    df.reset_index(drop=True, inplace=True)
    norm_j = df['norm_j']
    plt.plot(df.index, norm_j, label='j', linewidth=15)
    norm_bar = df['norm_bar']
    barlist=plt.bar(df.index, norm_bar)
    for i in range(len(df.index)):
        if norm_bar[i] <= 0:
            barlist[i].set_color('r')
        else:
            barlist[i].set_color('y')
    plt.axis('off')
    max_j = norm_j.abs().max()
    max_macd = norm_bar.abs().max()
    y_max = max(max_j, max_macd)
    plt.ylim((0-y_max)*1.1, y_max*1.1)

    str_symbol = symbol[:-3]
    if df['close_five'][2] >= 0.05:
        fig.savefig('./data/prepared/pic_data/macd_j_pic/up/' + str_symbol + '_' + str(sequence), dpi=20)
    elif df['close_five'][2] <= -0.05:
        fig.savefig('./data/prepared/pic_data/macd_j_pic/down/' + str_symbol + '_' + str(sequence), dpi=20)
    else:
        fig.savefig('./data/prepared/pic_data/macd_j_pic/equal/' + str_symbol + '_' + str(sequence), dpi=20)
    plt.close()

    return 1


def draw_pic(symbol, pic_type=None):
    df = pd.read_csv('./data/prepared/datacsv/' + symbol + '.csv', sep=',')
    df.index = pd.to_datetime(df['trade_date'], format='%Y/%m/%d')
    # df = df['2017-11-01':'2017-12-31']
    df.reset_index(drop = True, inplace = True)
    pic_count = 0
    for i in range(len(df) - 8):
        if pic_type == 'kdj':
            pic_res = draw_kdj_pic(symbol, df[0 + i:3 + i], i)  # 画图的
            if pic_res == 1:
                pic_count += 1
        elif pic_type == 'macd':
            pic_res = create_macd_pic(symbol, df[0 + i:3 + i], i)  # 画图的
            if pic_res == 1:
                pic_count += 1
        elif pic_type == 'fix':
            pic_res = create_fixed_pic(symbol, df[0 + i:3 + i], i)
            if pic_res == 1:
                pic_count += 1
        elif pic_type == 'macd_j':
            j = df['j']
            df['norm_j'] = j.apply(lambda x: (x - j.mean()) / (j.std()))
            bar_value = df['MACDhist']*2
            df['norm_bar'] = bar_value.apply(lambda x: (x - bar_value.mean()) / (bar_value.std()))
            pic_res = create_macd_j_pic(symbol, df[0 + i:3 + i], i)
            if pic_res == 1:
                pic_count += 1
        else:
            pic_res_kdj = draw_kdj_pic(symbol, df[0 + i:3 + i], i)
            pic_res_macd = create_macd_pic(symbol, df[0 + i:3 + i], i)
            if pic_res_kdj == 1:
                pic_count += 1
            if pic_res_macd == 1:
                pic_count += 1

    print("stock:" + symbol + ';画图数量：' + str(pic_count))


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


# get_data()
# draw_pic('600703.SH', 'macd_j')
# choice_pics('./data/prepared/pic_data/macd_j_pic2/')
# download_data()
# save_data_to_csv()

# df = pd.read_csv('./read_picture/data/csv/600030.SH.csv', sep=',')
df = pd.read_csv('./read_picture/data/csv/000012.SZ.csv', sep=',')
df = df[(df[['volume']] != 0).all(axis=1)]
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df = df[df['date'] > pd.datetime(2018, 10, 31)]
draw_indictors.plot_all(df, is_show=False, output='./read_picture/data/img/001.png')
