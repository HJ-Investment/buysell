# encoding: utf-8

from jaqs.data import RemoteDataService, DataView
import tushare as ts
from jaqs.data import DataApi
import numpy as np
import pandas as pd
import talib
import os
import matplotlib.pyplot as plt

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

dataview_store_folder = './data/prepared'


def download_data():
    dataview_props = {'start_date': 20080101, 'end_date': 20180731,
                      'universe': '000016.SH',
                      'fields': 'open,close,high,low',
                      'freq': 1}

    ds = RemoteDataService()
    ds.init_from_config(data_config)

    # DataView utilizes RemoteDataService to get various data and store them
    dv = DataView()
    dv.init_from_config(dataview_props, ds)
    dv.prepare_data()
    dv.save_dataview(folder_path=dataview_store_folder)


def load_data(symbol):
    dv = DataView()
    dv.load_dataview(folder_path=dataview_store_folder)

    df = pd.DataFrame()

    df['close'] = dv.get_ts('close', symbol=symbol, start_date=20080101, end_date=20171231)[symbol]
    df['open']  = dv.get_ts('open', symbol=symbol, start_date=20080101, end_date=20171231)[symbol]
    df['high']  = dv.get_ts('high', symbol=symbol, start_date=20080101, end_date=20171231)[symbol]
    df['low']   = dv.get_ts('low', symbol=symbol, start_date=20080101, end_date=20171231)[symbol]

    return df


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
    print(df)
    df['close_five'] = (df['close_five_value'] - df['close']) / df['close']
    return df


def save_csv(symbol, df):
    df.to_csv(path_or_buf='./data/prepared/datacsv/' + symbol + '.csv', sep=',', index=True)


def get_data(symbol=None):
    if not os.path.exists('./data/prepared/datacsv/'):
        os.makedirs('./data/prepared/datacsv/')

    if not symbol:
        api = DataApi(addr="tcp://data.quantos.org:8910")
        result, msg = api.login("18652420434", "eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MTcwNjAxMDgyOTMiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg2NTI0MjA0MzQifQ.b1ejSpbEVS7LhbsveZ5kvbWgUs7fnUd0-CBakPwNUu4")
        print(result)
        data, msg = api.query(
                view="lb.indexCons", 
                fields="symbol", 
                filter="index_code=000016.SH&start_date=20180801&end_date=20180831", 
                data_format='pandas')
        print(msg)
        symbols = data['symbol'].tolist()
        print(symbols)

        for sym in symbols:
            df = load_data(sym)
            close = [float(x) for x in df['close']]
            # prepare macd data
            df['MACD'], df['MACDsignal'], df['MACDhist'] = talib.MACD(np.array(close),
                                                                      fastperiod=12, slowperiod=26, signalperiod=9)
            df = df.sort_index()
            df.index = pd.to_datetime(df.index, format='%Y%m%d')
            df = prepare_kdj(df, 9, 'close')  # 计算好kdj之后从13行开始取数据,计算出来的kdj比较准确
            df = df[34:]
            df = calculator_close(df)
            save_csv(sym, df)

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
    print(df['close_five'][4])
    str_symbol = symbol[:-3]
    if df['close_five'][4] >= 0.05:
        fig.savefig('./data/prepared/pic_data/kdj_pic/up/' + str_symbol + '_' + str(sequence), dpi=20)
    elif df['close_five'][4] <= -0.05:
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
    plt.axis('off')

    str_symbol = symbol[:-3]
    if df['close_five'][4] >= 0.05:
        fig.savefig('./data/prepared/pic_data/macd_pic/up/' + str_symbol + '_' + str(sequence), dpi=20)
    elif df['close_five'][4] <= -0.05:
        fig.savefig('./data/prepared/pic_data/macd_pic/down/' + str_symbol + '_' + str(sequence), dpi=20)
    else:
        fig.savefig('./data/prepared/pic_data/macd_pic/equal/' + str_symbol + '_' + str(sequence), dpi=20)
    plt.close()

    return 1


def draw_pic(symbol, pic_type=None):
    df = pd.read_csv('./data/prepared/datacsv/' + symbol + '.csv', sep=',')
    df.index = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    # df = df['2017-01-01':'2017-12-31']
    pic_count = 0
    for i in range(len(df) - 10):
        if pic_type == 'kdj':
            pic_res = draw_kdj_pic(symbol, df[0 + i:5 + i], i)  # 画图的
            if pic_res == 1:
                pic_count += 1
        elif pic_type == 'macd':
            pic_res = create_macd_pic(symbol, df[0 + i:5 + i], i)  # 画图的
            if pic_res == 1:
                pic_count += 1
        else:
            pic_res_kdj = draw_kdj_pic(symbol, df[0 + i:5 + i], i)
            pic_res_macd = create_macd_pic(symbol, df[0 + i:5 + i], i)
            if pic_res_kdj == 1:
                pic_count += 1
            if pic_res_macd == 1:
                pic_count += 1

    print("stock:" + symbol + ';画图数量：' + str(pic_count))


# get_data('600000.SH')
draw_pic('600000.SH')