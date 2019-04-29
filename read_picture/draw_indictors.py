# encoding: utf-8
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
from pylab import rcParams
import numpy as np
from matplotlib import dates as mdates
import matplotlib.ticker as ticker

def plot_all(data, is_show=True, output=None):
    rcParams['figure.figsize'] = (12, 6)

    plt.figure()

    data = data.dropna()

    ax_k = plt.subplot(4, 2, 1)
    quotes = data[['date','open','high','low','close']]
    date_tickers=quotes.date.dt.date.values
    #用mdate产生连续时间
    # quotes['date'] = mdates.date2num(quotes['date'])
    # ax_k.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #我们要去掉周末，停牌，就不能传连续时间
    quotes['date'] = range(len(quotes))
    candlestick_ohlc(ax_k, quotes.values, colordown='#53c156', colorup='#ff1717', width=0.5)

    def format_date(x,pos=None):
        if x<0 or x>len(date_tickers)-1:
            return ''
        return date_tickers[int(x)]
    # ax_k.xaxis.set_major_locator(ticker.MultipleLocator(6))
    ax_k.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    plt.setp(ax_k.get_xticklabels(), fontsize=5)
    # plt.title("K线图")
    plt.axis('off')

    #volume
    ax0 = plt.subplot(4, 2, 2)
    plt.fill_between(data['date'], data['volume'], 0, label='volume')
    # plt.legend()
    plt.setp(ax0.get_xticklabels(), visible=False)
    # ax0.axes.yaxis.set_ticklabels([])
    # ax0.set_ylim(0, 3*data.volume.values.max())
    plt.axis('off')

    #cci
    ax1 = plt.subplot(4, 2, 3)
    plt.plot(data['date'], data['cci'], label='cci')
    # plt.title("cci")
    # plt.legend()
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.axis('off')

    # kdj
    ax2 = plt.subplot(4, 2, 4)
    plt.plot(data["date"], data["k"], label="K", color='gray')
    plt.plot(data["date"], data["d"], label="D", color='yellow')
    plt.plot(data["date"], data["j"], label="J", color='blue')
    # plt.title("KDJ")
    # plt.xlabel('date')
    # plt.ylabel('value')
    # plt.legend()
    plt.setp(ax2.get_xticklabels(), visible=False)
    # plt.xticks(rotation=90)
    plt.axis('off')

    # macd
    ax3 = plt.subplot(4, 2, 5)
    OSC = data['macd']
    DIFF = data['diff']
    DEM = data['dea']
    plt.fill_between(data["date"], OSC, 0, label="OSC")
    plt.plot(data["date"], DIFF, label="DIFF")
    plt.plot(data["date"], DEM, label="DEM")
    # plt.title("MACD")
    # plt.xlabel('date')
    # plt.ylabel('value')
    # plt.legend()
    plt.setp(ax3.get_xticklabels(), visible=False)
    # plt.xticks(rotation=90)
    plt.axis('off')

    #rsi
    ax4 = plt.subplot(4, 2, 6)
    # RSI6 = rsi(data, 6)
    # RSI12 = rsi(data, 12)
    # RSI24 = rsi(data, 24)
    plt.plot(data["date"], data['rsi'], label="RSI(n=6)")
    # plt.plot(RSI12["date"], RSI12['rsi'], label="RSI(n=12)")
    # plt.plot(RSI24["date"], RSI24['rsi'], label="RSI(n=24)")
    # plt.title("RSI")
    # plt.xlabel('date')
    # plt.ylabel('value')
    # plt.legend()
    # plt.xticks(rotation=90)
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.axis('off')

    #_boll
    ax5 = plt.subplot(4, 2, 7)
    plt.plot(data["date"], data['boll_mid'], label="BOLL(n=10)")
    plt.plot(data["date"], data['boll_up'], label="UPPER(n=10)")
    plt.plot(data["date"], data['boll_low'], label="LOWER(n=10)")
    plt.plot(data["date"], data["close"], label="CLOSE PRICE")
    # plt.title("BOLL")
    # plt.xlabel('date')
    # plt.ylabel('value')
    # plt.legend()
    # plt.xticks(rotation=90)
    plt.setp(ax5.get_xticklabels(), fontsize=3)
    plt.axis('off')

    # wr
    ax6 = plt.subplot(4, 2, 8)
    plt.plot(data['date'], data['wr'], label='wr')
    # plt.title("wr")
    # plt.legend()
    plt.setp(ax6.get_xticklabels(), fontsize=3)
    plt.axis('off')

    plt.tight_layout()

    if is_show:
        plt.show()

    if output is not None:
        plt.savefig(output)
        plt.close('all')