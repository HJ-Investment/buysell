# import matplotlib
#
# import tushare as ts
#
# import pandas as pd
#
# import matplotlib.pyplot as plt
#
# fig=plt.gcf()
#
# df=ts.get_hist_data('000001',start='2015-11-01',end='2015-12-31')
#
# with pd.plotting.plot_params.use('x_compat',True):
#     df.high.plot(color='r',figsize=(10,4),grid='on')
#     df.low.plot(color='b',figsize=(10,4),grid='on')
#     fig.savefig('E:\\Python')

#######以上是画分时图


import matplotlib.pyplot as plt
import numpy as np
import talib
import tushare as ts
from numpy import nan as NaN
import pandas as pd

fig=plt.gcf()
df=ts.get_k_data('600000',start='2015-01-01', end='2018-05-03',)
close = [float(x) for x in df['close']]
# 调用talib计算指数移动平均线的值
df['EMA12'] = talib.EMA(np.array(close), timeperiod=12)
df['EMA26'] = talib.EMA(np.array(close), timeperiod=26)
 # 调用talib计算MACD指标,返回的MACD是DIF，快线；MACDsignal是DEA值，慢线；macd=dif-DEA画成柱状图
df['MACD'],df['MACDsignal'],df['MACDhist'] = talib.MACD(np.array(close),
                            fastperiod=12, slowperiod=26, signalperiod=9)

df=df.fillna(method='bfill')
# df.to_csv('E://stock_data.csv', index=True,sep='\t', encoding='utf-8')

##talib返回的MACD是DIF，快线；MACDsignal是DEA值，慢线；macd=dif-DEA画成柱状图
macd_df = df[["date","MACD","MACDsignal","MACDhist"]]
# print(macd_df.head(50))


print(macd_df.head(50))
# macd_df.date=pd.to_datetime(macd_df.date)
# macd_df_date = macd_df.set_index("date")
macd_df_date_60 = macd_df.tail(60)
# print(macd_df_date_60)



# with pd.plotting.plot_params.use('x_compat',True):
#     plt.xticks(rotation=20)
#     plt.grid=True
#     # macd_df_date_60.MACD.plot(color='r',figsize=(10,4),grid='on')#快线
#     plt.plot(macd_df_date_60.index, macd_df_date_60['MACD'], label='macd dif')#快线
#     plt.plot(macd_df_date_60.index, macd_df_date_60['MACDsignal'], label='signal dea')# 慢线
#     plt.bar(macd_df_date_60.index, macd_df_date_60['MACDhist']*2, label='hist bar')
#     fig.savefig('E:\\MACD001')


