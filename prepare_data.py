# encoding: utf-8

from jaqs.data import RemoteDataService, DataView
import tushare as ts
from jaqs.data import DataApi
import pandas as pd
import talib

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
    dataview_props = {'start_date': 20160101, 'end_date': 20180731,
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

    df['close'] = dv.get_ts('close', symbol=symbol, start_date=20160101, end_date=20180430)['600016.SH']
    df['open']  = dv.get_ts('open', symbol=symbol, start_date=20160101, end_date=20180430)['600016.SH']
    df['high']  = dv.get_ts('high', symbol=symbol, start_date=20160101, end_date=20180430)['600016.SH']
    df['low']   = dv.get_ts('low', symbol=symbol, start_date=20160101, end_date=20180430)['600016.SH']
    
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
    low_list = pd.rolling_min(df['low'], n)
    low_list.fillna(value=pd.expanding_min(df['low']), inplace=True)
    high_list = pd.rolling_max(df['high'], n)
    high_list.fillna(value=pd.expanding_max(df['high']), inplace=True)
    rsv = (df[ksgn] - low_list) / (high_list - low_list) * 100

    df['k'] = pd.ewma(rsv, com=2)
    df['d'] = pd.ewma(df['k'], com=2)
    df['j'] = 3.0 * df['k'] - 2.0 * df['d']
    # print('n df',len(df))
    return df


def get_data(symbol=None):
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
            df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
            df = prepare_kdj(df, 9, 'close')  # 计算好kdj之后从13行开始取数据,计算出来的kdj比较准确
            df = df[34:]
            df.to_csv(path_or_buf='F:\Code\\buysell\data\pic_data\datacsv\\' + symbol + '.csv', sep=',', index=True)

    else:
        df = load_data(symbol)
        close = [float(x) for x in df['close']]
        # prepare macd data
        df['MACD'], df['MACDsignal'], df['MACDhist'] = talib.MACD(np.array(close),
                                                                  fastperiod=12, slowperiod=26, signalperiod=9)
        df = df.sort_index()
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
        df = prepare_kdj(df, 9, 'close')  # 计算好kdj之后从13行开始取数据,计算出来的kdj比较准确
        df = df[34:]
        df.to_csv(path_or_buf='F:\Code\\buysell\data\pic_data\datacsv\\' + symbol + '.csv', sep=',', index=True)


get_data()
