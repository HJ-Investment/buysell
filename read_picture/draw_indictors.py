# encoding: utf-8


def plot_all(data, is_show=True, output=None):
    import matplotlib.pyplot as plt
    from mpl_finance import candlestick_ohlc
    from pylab import rcParams
    import numpy as np
    from matplotlib import dates as mdates
    rcParams['figure.figsize'] = (12, 6)

    plt.figure()

    data = data.dropna()

    ax0 = plt.subplot(3, 1, 1)
    quotes = data[['date','open','high','low','close']]
    quotes['date'] = mdates.date2num(quotes['date'])
    candlestick_ohlc(ax0, quotes.values, colordown='#53c156', colorup='#ff1717', width=0.5)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.legend()
    plt.setp(ax0.get_xticklabels(), fontsize=5)

    # kdj
    ax1 = plt.subplot(3, 2, 3)
    plt.plot(data["date"], data["k"], label="K", color='gray')
    plt.plot(data["date"], data["d"], label="D", color='yellow')
    plt.plot(data["date"], data["j"], label="J", color='blue')
    plt.title("KDJ")
    # plt.xlabel('date')
    # plt.ylabel('value')
    plt.legend()
    plt.setp(ax1.get_xticklabels(), visible=False)
    # plt.xticks(rotation=90)

    # macd
    ax2 = plt.subplot(3, 2, 4)
    OSC = data['macd']
    DIFF = data['diff']
    DEM = data['dea']
    plt.fill_between(data["date"], OSC, 0, label="OSC")
    plt.plot(data["date"], DIFF, label="DIFF")
    plt.plot(data["date"], DEM, label="DEM")
    plt.title("MACD")
    # plt.xlabel('date')
    # plt.ylabel('value')
    plt.legend()
    plt.setp(ax2.get_xticklabels(), visible=False)
    # plt.xticks(rotation=90)

    #rsi
    ax3 = plt.subplot(3, 2, 5)
    # RSI6 = rsi(data, 6)
    # RSI12 = rsi(data, 12)
    # RSI24 = rsi(data, 24)
    plt.plot(data["date"], data['rsi'], label="RSI(n=6)")
    # plt.plot(RSI12["date"], RSI12['rsi'], label="RSI(n=12)")
    # plt.plot(RSI24["date"], RSI24['rsi'], label="RSI(n=24)")
    plt.title("RSI")
    # plt.xlabel('date')
    # plt.ylabel('value')
    plt.legend()
    # plt.xticks(rotation=90)
    plt.setp(ax3.get_xticklabels(), fontsize=3)

    #_boll
    ax4 = plt.subplot(3, 2, 6)
    plt.plot(data["date"], data['boll_mid'], label="BOLL(n=10)")
    plt.plot(data["date"], data['boll_up'], label="UPPER(n=10)")
    plt.plot(data["date"], data['boll_low'], label="LOWER(n=10)")
    plt.plot(data["date"], data["close"], label="CLOSE PRICE")
    plt.title("BOLL")
    # plt.xlabel('date')
    # plt.ylabel('value')
    plt.legend()
    # plt.xticks(rotation=90)
    plt.setp(ax4.get_xticklabels(), fontsize=3)

    plt.tight_layout()

    if is_show:
        plt.show()

    if output is not None:
        plt.savefig(output)