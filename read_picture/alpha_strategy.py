# encoding: utf-8

"""
A very first example of AlphaStrategy back-test:
    Market value weight among UNIVERSE.
    Benchmark is HS300.
    
"""
from __future__ import print_function, unicode_literals, division, absolute_import

from jaqs.data import RemoteDataService, DataView

import jaqs.util as jutil

from jaqs.trade import model
from jaqs.trade import (AlphaStrategy, AlphaBacktestInstance, AlphaTradeApi,
                        PortfolioManager, AlphaLiveTradeInstance, RealTimeTradeApi)
import jaqs.trade.analyze as ana

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

# Data files are stored in this folder:
dataview_store_folder = './output/alpha/prepare/dataview'

# Back-test and analysis results are stored here
backtest_result_folder = './output/alpha/HTML'

UNIVERSE = '000905.SH'


def save_data():
    """
    This function fetches data from remote server and stores them locally.
    Then we can use local data to do back-test.
    """
    dataview_props = {'start_date': 20170101,  # Start and end date of back-test
                      'end_date': 20171030,
                      'universe': UNIVERSE,    # Investment universe and performance benchmark
                      'benchmark': '000905.SH',
                      'fields': 'high,low,close', # Data fields that we need
                      'freq': 1   # freq = 1 means we use daily data. Please do not change this.
                      }

    # RemoteDataService communicates with a remote server to fetch data
    ds = RemoteDataService()

    # Use username and password in data_config to login
    ds.init_from_config(data_config)
    
    # DataView utilizes RemoteDataService to get various data and store them
    dv = DataView()
    dv.init_from_config(dataview_props, ds)
    dv.prepare_data()

    # 以9日为周期的KD线为例。首先须计算出最近9日的RSV值,即未成熟随机值, 
    # 计算公式为 
    # 9日RSV=(C-L9)÷(H9-L9)×100 
    # 式中,C为第9日的收盘价;L9为9日内的最低价;H9为9日内的最高价。 
    # K值=2/3×前一日K值+1/3×当日RSV 
    # D值=2/3×前一日K值+1/3×当日RSV 
    # J指标的计算公式为: J=3D—2K 
    factor_formula = '2/3*50 + (close-Ts_Min(low，5))/(Ts_Max(high，5)-Ts_Min(low，5)*100)'
    dv.add_formula()

    dv.save_dataview(folder_path=dataview_store_folder)


def create_macd_j_pic(self, symbol, df, sequence):
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
        max_j = norm_j.abs().max()
        max_macd = norm_bar.abs().max()
        y_max = max(max_j, max_macd)
        plt.ylim((0-y_max)*1.1, y_max*1.1)

        str_symbol = symbol[:-3]
        pic_path = './output/rnn_50/pic_data/' + str_symbol + '_' + str(sequence) + '.jpg'
        fig.savefig(pic_path, dpi=20)
        plt.close()

        return pic_path

def predict(self, quote):
    logger.info('---------------------------PREDICT---------------------------------')
    logger.info(quote.symbol)
    df = self.price_arr[quote.symbol]
    low_list = df['low'].rolling(window=9, center=False).min()  # pd.rolling_min(df['low'], n)
    low_list.fillna(value=df['low'].expanding(min_periods=1).min(), inplace=True)
    high_list = df['high'].rolling(window=9, center=False).max()  # pd.rolling_max(df['high'], n)
    high_list.fillna(value=df['high'].expanding(min_periods=1).max(), inplace=True)
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100

    df['k'] = pd.ewma(rsv, com=2)
    df['d'] = pd.ewma(df['k'], com=2)
    df['j'] = 3.0 * df['k'] - 2.0 * df['d']

    close = [float(x) for x in df['close']]
    # prepare macd data
    df['MACD'], df['MACDsignal'], df['MACDhist'] = talib.MACD(np.array(close),
                                                            fastperiod=12, slowperiod=26, signalperiod=9)

    
    #上新
    if len(df.index) < 42:
        return
    # 停牌
    if df.iloc[-3,0] == 0 or df.iloc[-2,0] == 0 or df.iloc[-1,0] == 0:
        return 0
    j = df['j']
    df['norm_j'] = j.apply(lambda x: (x - j.mean()) / (j.std()))
    bar_value = df['MACDhist']*2
    
    # print(df)
    df['norm_bar'] = bar_value.apply(lambda x: (x - bar_value.mean()) / (bar_value.std()))
    pic_path = self.create_macd_j_pic(quote.symbol, df[-3:], self.window_count)

    result = run_inference_on_image(pic_path)
    logger.info(result)
    logger.info('-------------------------------------------------------------------\n')
    return result

def do_backtest():
    # Load local data file that we just stored.
    dv = DataView()
    dv.load_dataview(folder_path=dataview_store_folder)
    
    backtest_props = {"start_date"      : dv.start_date, # start and end date of back-test
                      "end_date"        : dv.end_date,
                      "period"          : "month",           # re-balance period length
                      "benchmark"       : dv.benchmark,   # benchmark and universe
                      "universe"        : dv.universe,
                      "init_balance"    : 1e8,         # Amount of money at the start of back-test
                      "position_ratio"  : 1.0,       # Amount of money at the start of back-test
                      }
    backtest_props.update(data_config)
    backtest_props.update(trade_config)

    # Create model context using AlphaTradeApi, AlphaStrategy, PortfolioManager and AlphaBacktestInstance.
    # We can store anything, e.g., public variables in context.

    trade_api = AlphaTradeApi()
    strategy = AlphaStrategy(pc_method='market_value_weight')
    pm = PortfolioManager()
    bt = AlphaBacktestInstance()
    context = model.Context(dataview=dv, instance=bt, strategy=strategy, trade_api=trade_api, pm=pm)

    bt.init_from_config(backtest_props)
    bt.run_alpha()

    # After finishing back-test, we save trade results into a folder
    bt.save_results(folder_path=backtest_result_folder)


def do_livetrade():
    dv = DataView()
    dv.load_dataview(folder_path=dataview_store_folder)
    
    props = {"period": "day",
             "strategy_no": 1044,
             "init_balance": 1e6}
    props.update(data_config)
    props.update(trade_config)
    
    strategy = AlphaStrategy(pc_method='market_value_weight')
    pm = PortfolioManager()
    
    bt = AlphaLiveTradeInstance()
    trade_api = RealTimeTradeApi(props)
    ds = RemoteDataService()
    
    context = model.Context(dataview=dv, instance=bt, strategy=strategy, trade_api=trade_api, pm=pm, data_api=ds)
    
    bt.init_from_config(props)
    bt.run_alpha()
    
    goal_positions = strategy.goal_positions
    print("Length of goal positions:", len(goal_positions))
    task_id, msg = trade_api.goal_portfolio(goal_positions)
    print(task_id, msg)


def analyze_backtest_results():
    # Analyzer help us calculate various trade statistics according to trade results.
    # All the calculation results will be stored as its members.
    ta = ana.AlphaAnalyzer()
    dv = DataView()
    dv.load_dataview(folder_path=dataview_store_folder)
    
    ta.initialize(dataview=dv, file_folder=backtest_result_folder)

    ta.do_analyze(result_dir=backtest_result_folder,
                  selected_sec=list(ta.universe)[:3])


if __name__ == "__main__":
    is_backtest = True
    
    if is_backtest:
        save_data()
        do_backtest()
        analyze_backtest_results()
    else:
        save_data()
        do_livetrade()