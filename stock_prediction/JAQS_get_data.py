# import tushare as ts
# df=ts.get_k_data('002743')
# # print(df)
# print(type(df))
####jaqs里的dataview测试

from __future__ import print_function, unicode_literals, division, absolute_import

from jaqs.data import RemoteDataService, DataView

import jaqs.util as jutil

from jaqs.trade import model
from jaqs.trade import (AlphaStrategy, AlphaBacktestInstance, AlphaTradeApi,
                        PortfolioManager, AlphaLiveTradeInstance, RealTimeTradeApi)
import jaqs.trade.analyze as ana

data_config = {
    "remote.data.address": "tcp://data.tushare.org:8910",
    "remote.data.username": "18652420434",
    "remote.data.password": "eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MTcwNjAxMDgyOTMiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg2NTI0MjA0MzQifQ.b1ejSpbEVS7LhbsveZ5kvbWgUs7fnUd0-CBakPwNUu4"
}
trade_config = {
    "remote.trade.address": "tcp://gw.quantos.org:8901",
    "remote.trade.username": "18652420434",
    "remote.trade.password": "eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MTcwNjAxMDgyOTMiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg2NTI0MjA0MzQifQ.b1ejSpbEVS7LhbsveZ5kvbWgUs7fnUd0-CBakPwNUu4"
}

# Data files are stored in this folder:
dataview_store_folder = '/dataview'
# print(dataview_store_folder)
# Back-test and analysis results are stored here
backtest_result_folder = '../../output/simplest'

UNIVERSE = '000807.SH'


def save_data():
    """
    This function fetches data from remote server and stores them locally.
    Then we can use local data to do back-test.
    """
    dataview_props = {'start_date': 20170101,  # Start and end date of back-test
                      'end_date': 20171030,
                      'symbol':'000300.SH',
                      # 'universe': UNIVERSE,  # Investment universe and performance benchmark
                      # 'benchmark': '000300.SH',
                      'fields': 'open,high,low,close,volume',  # Data fields that we need
                      'freq': 1  # freq = 1 means we use daily data. Please do not change this.
                      }

    # RemoteDataService communicates with a remote server to fetch data
    ds = RemoteDataService()

    # Use username and password in data_config to login
    ds.init_from_config(data_config)

    # DataView utilizes RemoteDataService to get various data and store them
    dv = DataView()
    dv.init_from_config(dataview_props, ds)
    dv.prepare_data()
    print(dv.prepare_data())
    dv.save_dataview(folder_path=dataview_store_folder)

# save_data()



dv = DataView()
dv.load_dataview('/dataview')
df = dv.get_ts('open,close,high,low',symbol='000300.SH',start_date=20170101, end_date=20170302)
df.columns = ['open', 'close', 'high', 'low']
print(df)