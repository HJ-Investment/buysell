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

start_date = "20120101"
end_date = "20181231"

def get_index_info():
    df, msg = api.query(
                view="lb.indexCons", 
                fields="index_code,symbol,in_date,out_date", 
                filter="index_code=000905.SH&start_date=20120101&end_date=20181231", 
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
        ts_symbol = dv.get_ts('open,close,high,low,volume,future_return_2,future_return_3,future_return_4,future_return_5', symbol=symbol, start_date=start_date, end_date=end_date)[symbol]

        ts_symbol.fillna(0, inplace=True)
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
    

    # download_data()
    # save_data_to_csv()

    count = 0

    # sh_000905 = {'symbol':['000950.SZ']}
    # sh_000905 = get_index_info()
    sh_000905 = {'symbol':[ '000729.SZ',
                            '000732.SZ', '000766.SZ', '000778.SZ', '002120.SZ', '002127.SZ', '002147.SZ', '002468.SZ', '002517.SZ', '002583.SZ', '002601.SZ', '002709.SZ', '002807.SZ', '002815.SZ', '002818.SZ', '300002.SZ', '300058.SZ', '300116.SZ', '300146.SZ', '300376.SZ', '300383.SZ', '600126.SH', '600566.SH', '600582.SH', '600642.SH', '600648.SH', '600754.SH', '600839.SH', '600936.SH', '600996.SH', '601020.SH', '601811.SH', '601928.SH', '603000.SH', '603228.SH', '603444.SH', '603515.SH', '603556.SH', '603569.SH', '603658.SH', '603799.SH', '603816.SH', '603877.SH', '603888.SH', '000961.SZ', '600864.SH', '000900.SZ', '000788.SZ', '601886.SH', '600231.SH', '600884.SH', '000410.SZ', '603188.SH', '002589.SZ', '000949.SZ', '600227.SH', '002029.SZ', '002568.SZ', '600178.SH', '600293.SH', '600971.SH', '600496.SH', '600983.SH', '000837.SZ', '600844.SH', '600851.SH', '600289.SH', '002308.SZ', '000875.SZ', '600467.SH', '002384.SZ', '600325.SH', '600885.SH', '002238.SZ', '002714.SZ', '002368.SZ', '600707.SH', '000008.SZ', '000553.SZ', '000686.SZ', '000738.SZ', '000750.SZ',
                            '000813.SZ', '000980.SZ', '002110.SZ', '002129.SZ', '002174.SZ', '002180.SZ', '002212.SZ', '002302.SZ', '002399.SZ', '002424.SZ', '002426.SZ', '002465.SZ', '002509.SZ', '002563.SZ', '002600.SZ', '002831.SZ', '002839.SZ', '002841.SZ', '002916.SZ', '002920.SZ', '300156.SZ', '300308.SZ', '300315.SZ', '300418.SZ', '600021.SH', '600167.SH', '600258.SH', '600350.SH', '600528.SH', '600649.SH', '600760.SH', '600777.SH', '600827.SH', '600895.SH', '601019.SH', '601098.SH', '601179.SH', '601608.SH', '601801.SH', '601872.SH', '601966.SH', '603056.SH', '603659.SH', '603885.SH', '603899.SH', '600637.SH', '600498.SH', '002378.SZ', '002051.SZ', '600022.SH', '002075.SZ', '002410.SZ', '002106.SZ', '600872.SH', '600675.SH', '002390.SZ', '000810.SZ', '000506.SZ', '002396.SZ', '600497.SH', '000407.SZ', '300001.SZ', '600664.SH', '000825.SZ', '600507.SH', '600329.SH', '600572.SH', '600006.SH', '000917.SZ', '600797.SH', '600323.SH', '600388.SH', '600073.SH', '600546.SH', '601886.SH', '000937.SZ', '601311.SH', '600699.SH', '600112.SH', '002612.SZ', '000723.SZ',
                            '603001.SH', '002477.SZ', '002585.SZ', '600790.SH', '600175.SH', '002093.SZ', '002408.SZ', '600348.SH', '000973.SZ', '002638.SZ', '000671.SZ', '000088.SZ', '002662.SZ', '601588.SH', '000543.SZ', '600006.SH', '300324.SZ', '002128.SZ', '600651.SH', '600061.SH', '600517.SH', '300026.SZ', '000422.SZ', '601801.SH', '002727.SZ', '002191.SZ', '000598.SZ', '600195.SH', '601880.SH', '600503.SH', '002232.SZ', '603568.SH', '000571.SZ', '002440.SZ', '600435.SH', '600499.SH', '600428.SH', '600246.SH', '002049.SZ', '600614.SH', '002456.SZ', '601001.SH', '000572.SZ', '002019.SZ', '000897.SZ', '002052.SZ', '600198.SH', '600616.SH', '002508.SZ', '600456.SH', '600422.SH', '000830.SZ', '600087.SH', '002310.SZ', '600432.SH', '002065.SZ', '000708.SZ', '000608.SZ', '300291.SZ', '601011.SH', '601588.SH', '002073.SZ', '000959.SZ', '002690.SZ', '600724.SH', '002470.SZ', '603169.SH', '600458.SH', '002225.SZ', '603366.SH', '601908.SH', '600300.SH', '601107.SH', '002283.SZ', '000998.SZ', '600429.SH', '002168.SZ', '000581.SZ', '000899.SZ', '002745.SZ', '600280.SH',
                            '000510.SZ', '600376.SH', '600260.SH', '600600.SH', '600531.SH', '600623.SH', '600143.SH', '002480.SZ', '300033.SZ', '600866.SH', '000415.SZ', '000935.SZ', '600717.SH', '000513.SZ', '600682.SH', '002285.SZ', '600403.SH', '600176.SH', '600359.SH', '000009.SZ', '000156.SZ', '000718.SZ', '000807.SZ', '000887.SZ', '000977.SZ', '000981.SZ', '000987.SZ', '000990.SZ', '002032.SZ', '002049.SZ', '002131.SZ', '002152.SZ', '002183.SZ', '002195.SZ', '002280.SZ', '002299.SZ', '002359.SZ', '002434.SZ', '002506.SZ', '002745.SZ', '300133.SZ', '300182.SZ', '300197.SZ', '600004.SH', '600037.SH', '600060.SH', '600155.SH', '600179.SH', '600256.SH', '600317.SH', '600346.SH', '600393.SH', '600438.SH', '600575.SH', '600578.SH', '600598.SH', '600623.SH', '600718.SH', '600737.SH', '600823.SH', '600875.SH', '600885.SH', '600908.SH', '600939.SH', '601127.SH', '601128.SH', '601200.SH', '601969.SH', '603225.SH', '600261.SH', '002204.SZ', '600363.SH', '002251.SZ', '600290.SH', '600268.SH', '600468.SH', '600966.SH', '600326.SH', '002180.SZ', '600575.SH', '600880.SH',
                            '002261.SZ', '601003.SH', '600259.SH', '600420.SH', '600351.SH', '000409.SZ', '002190.SZ', '603589.SH', '603019.SH', '600636.SH', '002657.SZ', '600717.SH', '002498.SZ', '002190.SZ', '000685.SZ', '600266.SH', '002393.SZ', '000603.SZ', '000799.SZ', '600823.SH', '000533.SZ', '002670.SZ', '600809.SH', '002078.SZ', '600570.SH', '600449.SH', '601608.SH', '600490.SH', '000829.SZ', '300113.SZ', '600641.SH', '600601.SH', '000989.SZ', '600220.SH', '600803.SH', '002110.SZ', '002505.SZ', '000031.SZ', '600587.SH', '000417.SZ', '002092.SZ', '002678.SZ', '600088.SH', '000861.SZ', '000748.SZ', '601678.SH', '600022.SH', '000963.SZ', '600160.SH', '600039.SH', '002038.SZ', '600054.SH', '000503.SZ', '600074.SH', '601233.SH', '000616.SZ', '600584.SH', '600418.SH', '000910.SZ', '002306.SZ', '600586.SH', '600867.SH', '002449.SZ', '600581.SH', '600673.SH', '600545.SH', '000158.SZ', '601101.SH', '002254.SZ', '002460.SZ', '002463.SZ', '600320.SH', '600525.SH', '300202.SZ', '002595.SZ', '600004.SH', '600480.SH', '600187.SH', '002416.SZ', '300199.SZ', '600806.SH',
                            '600612.SH', '002006.SZ', '600340.SH', '600801.SH', '600470.SH', '600103.SH', '600266.SH', '000690.SZ', '000869.SZ', '600223.SH', '000703.SZ', '600219.SH', '600058.SH', '600201.SH', '600582.SH', '002176.SZ', '600829.SH', '000927.SZ', '600594.SH', '000597.SZ', '600270.SH', '600633.SH', '600067.SH', '002482.SZ', '000777.SZ', '600863.SH', '600199.SH', '300059.SZ', '000021.SZ', '300297.SZ', '002273.SZ', '601369.SH', '000680.SZ', '600139.SH', '000620.SZ', '000631.SZ', '600386.SH', '000878.SZ', '002154.SZ', '002707.SZ', '600038.SH', '002064.SZ', '000713.SZ', '000030.SZ', '600322.SH', '002167.SZ', '600981.SH', '600287.SH', '600333.SH', '000999.SZ', '000541.SZ', '600516.SH', '601000.SH', '300144.SZ', '600644.SH', '002063.SZ', '000698.SZ', '002018.SZ', '600408.SH', '600654.SH', '002109.SZ', '000962.SZ', '600460.SH', '600098.SH', '600292.SH', '603198.SH', '000893.SZ', '300274.SZ', '600633.SH', '600169.SH', '600478.SH', '600997.SH', '000563.SZ', '600835.SH', '600963.SH', '601216.SH', '600765.SH', '000012.SZ', '600288.SH', '000667.SZ', '002444.SZ',
                            '600597.SH', '002383.SZ', '000877.SZ', '600740.SH', '600881.SH', '000809.SZ', '000525.SZ', '002028.SZ', '002327.SZ', '000932.SZ', '603355.SH', '000518.SZ', '000596.SZ', '600736.SH', '600171.SH', '002292.SZ', '002317.SZ', '000970.SZ', '600635.SH', '002475.SZ', '600433.SH', '300088.SZ', '000400.SZ', '002399.SZ', '000799.SZ', '600686.SH', '002491.SZ', '002512.SZ', '000977.SZ', '600210.SH', '600315.SH', '000988.SZ', '002353.SZ', '000806.SZ', '002311.SZ', '600284.SH', '000987.SZ', '600483.SH', '600522.SH', '600072.SH', '601689.SH', '000601.SZ', '600978.SH', '002244.SZ', '603868.SH', '600479.SH', '002041.SZ', '002701.SZ', '000650.SZ', '600508.SH', '000823.SZ', '000936.SZ', '600848.SH', '002429.SZ', '600754.SH', '600022.SH', '002574.SZ', '000617.SZ', '600216.SH', '000066.SZ', '600846.SH', '600917.SH', '000681.SZ', '600501.SH', '600060.SH', '002344.SZ', '000600.SZ', '002305.SZ', '600339.SH', '000982.SZ', '600086.SH', '600776.SH', '000972.SZ', '603005.SH', '000911.SZ', '000612.SZ', '002069.SZ', '002714.SZ', '601566.SH', '000488.SZ', '002366.SZ',
                            '600017.SH', '002422.SZ', '600704.SH', '600398.SH', '600895.SH', '600618.SH', '600648.SH', '600687.SH', '600436.SH', '600507.SH', '600078.SH', '600389.SH', '000931.SZ', '600729.SH', '000718.SZ', '600888.SH', '300166.SZ', '300156.SZ', '002013.SZ', '600874.SH', '600747.SH', '002276.SZ', '603077.SH', '002646.SZ', '600676.SH', '600094.SH', '601388.SH', '600826.SH', '600816.SH', '002430.SZ', '002226.SZ', '000930.SZ', '600611.SH', '002242.SZ', '600810.SH', '000517.SZ', '002646.SZ', '002640.SZ', '002018.SZ', '000426.SZ', '002648.SZ', '002573.SZ', '601929.SH', '300257.SZ', '002447.SZ', '603025.SH', '002048.SZ', '000029.SZ', '002223.SZ', '600487.SH', '600295.SH', '002424.SZ', '601519.SH', '600277.SH', '000816.SZ', '600277.SH', '600754.SH', '000951.SZ', '600269.SH', '603528.SH', '600685.SH', '600316.SH', '600387.SH', '000655.SZ', '600138.SH', '600815.SH', '000719.SZ', '000517.SZ', '002217.SZ', '600751.SH', '002479.SZ', '600993.SH', '600869.SH', '600392.SH', '600759.SH', '600773.SH', '300253.SZ', '600565.SH', '002392.SZ', '300267.SZ', '000060.SZ',
                            '000537.SZ', '000559.SZ', '000564.SZ', '000623.SZ', '000717.SZ', '000723.SZ', '000932.SZ', '000998.SZ', '002074.SZ', '002372.SZ', '002385.SZ', '002470.SZ', '002500.SZ', '002507.SZ', '300027.SZ', '300168.SZ', '300316.SZ', '300347.SZ', '300413.SZ', '300450.SZ', '300459.SZ', '600008.SH', '600373.SH', '600507.SH', '600567.SH', '600699.SH', '600779.SH', '600782.SH', '600804.SH', '600820.SH', '600901.SH', '600959.SH', '601003.SH', '601005.SH', '601100.SH', '601106.SH', '601231.SH', '601326.SH', '601718.SH', '601866.SH', '601869.SH', '601958.SH', '601990.SH', '603233.SH', '603486.SH', '603650.SH', '603712.SH', '603866.SH', '600811.SH', '002030.SZ', '000528.SZ', '000712.SZ', '002400.SZ', '600197.SH', '002001.SZ', '600827.SH', '002275.SZ', '000887.SZ', '600117.SH', '600110.SH', '600622.SH', '000050.SZ', '600683.SH', '002602.SZ', '600132.SH', '600360.SH', '603567.SH', '002122.SZ', '002153.SZ', '000970.SZ', '600397.SH', '002603.SZ', '600961.SH', '600079.SH', '601699.SH', '600122.SH', '002332.SZ', '600053.SH', '000627.SZ', '600528.SH', '002373.SZ',
                            '603555.SH', '000537.SZ', '000750.SZ', '600166.SH', '002663.SZ', '600511.SH', '000159.SZ', '000090.SZ', '300055.SZ', '600307.SH', '002181.SZ', '600877.SH', '600127.SH', '002050.SZ', '600858.SH', '002315.SZ', '600589.SH', '002430.SZ', '600743.SH', '002008.SZ', '600075.SH', '600750.SH', '002581.SZ', '600563.SH', '601002.SH', '600770.SH', '000050.SZ', '600720.SH', '600655.SH', '600064.SH', '002074.SZ', '601000.SH', '300122.SZ', '000550.SZ', '002320.SZ', '002466.SZ', '600416.SH', '002268.SZ', '002237.SZ', '601666.SH', '002269.SZ', '600995.SH', '600184.SH', '002287.SZ', '000598.SZ', '000925.SZ', '600021.SH', '002216.SZ', '000912.SZ', '600158.SH', '600623.SH', '601677.SH', '600239.SH', '600151.SH', '002172.SZ', '000587.SZ', '002049.SZ', '600569.SH', '600401.SH', '600523.SH', '600549.SH', '000755.SZ', '600388.SH', '000822.SZ', '601099.SH', '600640.SH', '000594.SZ', '600436.SH', '300010.SZ', '002400.SZ', '600566.SH', '000089.SZ', '002174.SZ', '002416.SZ', '600141.SH', '000918.SZ', '603698.SH', '600033.SH', '000959.SZ', '601139.SH', '300072.SZ',
                            '002672.SZ', '601208.SH', '600079.SH', '600595.SH', '002267.SZ', '601012.SH', '002498.SZ', '000860.SZ', '600291.SH', '000826.SZ', '600330.SH', '002143.SZ', '002097.SZ', '600596.SH', '002572.SZ', '002419.SZ', '600648.SH', '300182.SZ', '600825.SH', '002011.SZ', '600782.SH', '300159.SZ', '600162.SH', '600536.SH', '600308.SH', '600410.SH', '000511.SZ', '000850.SZ', '601231.SH', '000661.SZ', '600062.SH', '002439.SZ', '002681.SZ', '002551.SZ', '002698.SZ', '600537.SH', '000566.SZ', '600258.SH', '002262.SZ', '002642.SZ', '002493.SZ', '600488.SH', '600059.SH', '601616.SH', '002194.SZ', '000062.SZ', '601168.SH', '600202.SH', '601226.SH', '002665.SZ', '000751.SZ', '600880.SH', '600694.SH', '000786.SZ', '600757.SH', '601100.SH', '000428.SZ', '600602.SH', '600446.SH', '000921.SZ', '600702.SH', '002465.SZ', '300168.SZ', '000852.SZ', '600780.SH', '002601.SZ', '002195.SZ', '002032.SZ', '600503.SH', '603369.SH', '002336.SZ', '600628.SH', '600645.SH', '600481.SH', '600008.SH', '000762.SZ', '600120.SH', '600787.SH', '600566.SH', '000758.SZ', '002062.SZ',
                            '002004.SZ', '601872.SH', '600094.SH', '000028.SZ', '000301.SZ', '600894.SH', '002056.SZ', '600667.SH', '601929.SH', '000555.SZ', '600466.SH', '002506.SZ', '000592.SZ', '002002.SZ', '002371.SZ', 
                            '000687.SZ']}
    # print(sh_000905)
    for symbol in sh_000905['symbol']:
        start = time.time()
        print(symbol)
        count += 1
        print(str(count) + "/" + str(len(sh_000905)))
        df = pd.read_csv('./read_picture/data/csv/'+symbol+'.csv', sep=',')

        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

        # df.to_csv('./read_picture/data/csv/'+symbol+'_.csv', sep=',')

        if not os.path.exists('./read_picture/data/img/'+symbol+'/up'):
            os.makedirs('./read_picture/data/img/'+symbol+'/up')
        if not os.path.exists('./read_picture/data/img/'+symbol+'/down'):
            os.makedirs('./read_picture/data/img/'+symbol+'/down')
        
        ll = len(df) - 10 - 35
        multiprocessing.freeze_support()
        pool = multiprocessing.Pool()
        for i in range(ll):
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

