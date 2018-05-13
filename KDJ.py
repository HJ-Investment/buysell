import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import talib
from functools import reduce

def lm_kdj(df, n,ksgn='close'):
    '''
    【输入】
        df, pd.dataframe格式数据源
        n，时间长度
        ksgn，列名，一般是：close收盘价
    【输出】
        df, pd.dataframe格式数据源,
        增加了一栏：_{n}，输出数据
    '''
    lowList= pd.rolling_min(df['low'], n)
    lowList.fillna(value=pd.expanding_min(df['low']), inplace=True)
    highList = pd.rolling_max(df['high'], n)
    highList.fillna(value=pd.expanding_max(df['high']), inplace=True)
    rsv = (df[ksgn] - lowList) / (highList - lowList) * 100

    df['k'] = pd.ewma(rsv,com=2)
    df['d'] = pd.ewma(df['k'],com=2)
    df['j'] = 3.0 * df['k'] - 2.0 * df['d']
    #print('n df',len(df))
    return df




fig=plt.gcf()
df=ts.get_hist_data('600000',start='2016-12-22',end='2017-02-28')

df=df.sort_index()
df.index=pd.to_datetime(df.index,format='%Y-%m-%d')
#收市股价
close= df.close
highPrice=df.high
lowPrice=df.low
#每天的股价变动百分率
ret=df.p_change/100
 # 调用talib计算MACD指标
# df['k'],df['d']=talib.STOCH(np.array(highPrice),np.array(lowPrice),np.array(close),
#   fastk_period=9,slowk_period=3,slowk_matype=0,slowd_period=3,slowd_matype=0)
df=lm_kdj(df,9,'close')

df=df[13:]
# df=df.dropna(how='any')
sig_k=df.k
sig_d=df.d
sig_j=df.k*3-df.d*2

print(df)

plt.plot(sig_k.index,sig_k,label='k')
plt.plot(sig_d.index,sig_d,label='d')
plt.plot(sig_j.index,sig_j, label='j')
plt.show()
fig.savefig('E:\\KDJ001')









