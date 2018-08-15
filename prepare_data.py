# encoding: utf-8

from jaqs.data import RemoteDataService, DataView
import tushare as ts
from jaqs.data import DataApi
import pandas as pd

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

api = DataApi(addr="tcp://data.tushare.org:8910")
result, msg = api.login("18652420434", "eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MTcwNjAxMDgyOTMiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg2NTI0MjA0MzQifQ.b1ejSpbEVS7LhbsveZ5kvbWgUs7fnUd0-CBakPwNUu4")

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

    df['close'] = np.array(dv.get_ts('close', symbol=symbol, start_date=20160101, end_date=20180430)['close']
    df['open']  = dv.get_ts('open', symbol=symbol, start_date=20160101, end_date=20180430)['open']
    df['high']  = dv.get_ts('high', symbol=symbol, start_date=20160101, end_date=20180430)['high']
    df['low']   = dv.get_ts('low', symbol=symbol, start_date=20160101, end_date=20180430)['low']

    return df

def get_data(symbol=None):
    if not symbol:
        sz50s = ts.get_sz50s()
        one = sz50s['code'][1]



        print(one)
        one_symbol = load_data(one+'.SH')

        print(one_symbol)
    else:
        print('11')

get_data()
