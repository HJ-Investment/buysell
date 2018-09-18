from multiprocessing import Pool
from jaqs.data import DataApi
import pandas as pd


def create_macd_j_pic(symbol, df, sequence):
    import os
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    if not os.path.exists('./data/prepared/pic_data/macd_j_pic/up'):
        os.makedirs('./data/prepared/pic_data/macd_j_pic/up')
    if not os.path.exists('./data/prepared/pic_data/macd_j_pic/notup'):
        os.makedirs('./data/prepared/pic_data/macd_j_pic/notup')

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

    str_symbol = symbol[:-3]
    if df['close_five'][2] >= 0.05:
        fig.savefig('./data/prepared/pic_data/macd_j_pic/up/' + str_symbol + '_' + str(sequence), dpi=20)
    else:
        fig.savefig('./data/prepared/pic_data/macd_j_pic/notup/' + str_symbol + '_' + str(sequence), dpi=20)

    plt.close()

    print(sequence)

api = DataApi(addr="tcp://data.quantos.org:8910")
result, msg = api.login("18652420434", "eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MTcwNjAxMDgyOTMiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg2NTI0MjA0MzQifQ.b1ejSpbEVS7LhbsveZ5kvbWgUs7fnUd0-CBakPwNUu4")
# print(result)
data, msg = api.query(
        view="lb.indexCons",
        fields="symbol",
        filter="index_code=000016.SH&start_date=20180801&end_date=20180831",
        data_format='pandas')
print(msg)
symbols = data['symbol'].tolist()
print(symbols)

pool = Pool()
for sym in symbols:
    df = pd.read_csv('./data/prepared/datacsv/' + sym + '.csv', sep=',')
    df.index = pd.to_datetime(df['trade_date'], format='%Y/%m/%d')
    df.reset_index(drop=True, inplace=True)
    j = df['j']
    df['norm_j'] = j.apply(lambda x: (x - j.mean()) / (j.std()))
    bar_value = df['MACDhist'] * 2
    df['norm_bar'] = bar_value.apply(lambda x: (x - bar_value.mean()) / (bar_value.std()))

    for i in range(len(df) - 8):
        ts = df[0 + i:3 + i]
        pool.apply(create_macd_j_pic, args=(sym, ts, i))
