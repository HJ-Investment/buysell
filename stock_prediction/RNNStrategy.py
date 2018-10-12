# encoding: utf-8

from __future__ import print_function, division, unicode_literals, absolute_import

import time

import numpy as np
import pandas as pd
import talib

import os
import matplotlib.pyplot as plt

from jaqs.data import RemoteDataService
from jaqs.data.basic import Bar, Quote
from jaqs.trade import (model, EventLiveTradeInstance, EventBacktestInstance, RealTimeTradeApi,
                        EventDrivenStrategy, BacktestTradeApi, PortfolioManager, common)
import jaqs.trade.analyze as ana
import jaqs.util as jutil

from classify_image import run_inference_on_image, create_graph

import logging
from random import sample

logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件
fh = logging.FileHandler('./buysell.log', encoding='UTF-8')
fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

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

result_dir_path = './output/rnn_50'
start_date = 20180101
end_date = 20180930
index = '000016.SH'
is_backtest = True


class RNNStrategy(EventDrivenStrategy):
    """"""
    def __init__(self):
        super(RNNStrategy, self).__init__()

        # 标的
        self.symbol = ''
        self.balance = ''
        self.benchmark_symbol = ''
        self.quotelist = ''
        
        # 记录当前已经过的天数
        self.window_count = 0
        self.window = 0

        # 固定长度的价格序列
        self.price_arr = {}
        self.holding_day = {}

        # 当前仓位
        self.pos = {}
        self.holding_count = 0
        self.cur_holding_count = 0
        self.stock_value = 0

        # 下单量乘数
        self.buy_size_unit = 1
        self.output = True
    
    def init_from_config(self, props):
        """
        将props中的用户设置读入
        """
        super(RNNStrategy, self).init_from_config(props)
        # 标的
        self.symbol = props.get('symbol').split(',')
        self.benchmark_symbol = self.symbol[-1]

        # 初始资金
        self.init_balance = props.get('init_balance')
        # 最大持仓股票数
        self.holding_count = props.get('holding_Count')
        self.balance = self.init_balance
        # 每支股票购买资金
        self.stock_value = self.balance / self.holding_count

        self.window = 37
        
        # 固定长度的价格序列
        for s in self.symbol:
            self.price_arr[s] = pd.DataFrame(columns=['low', 'high', 'close'])
            self.pos[s] = 0
            self.holding_day[s] = 0
        create_graph('F:/Code/buysell/slim/macd_j/frozen_graph.pb')

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

    def buy(self, quote, size):
        """
        这里传入的'quote'可以是:
            - Quote类型 (在实盘/仿真交易和tick级回测中，为tick数据)
            - Bar类型 (在bar回测中，为分钟或日数据)
        我们通过isinsance()函数判断quote是Quote类型还是Bar类型
        """
        if isinstance(quote, Quote):
            # 如果是Quote类型，ref_price为bidprice和askprice的均值
            ref_price = (quote.bidprice1 + quote.askprice1) / 2.0
        else:
            # 否则为bar类型，ref_price为bar的收盘价
            ref_price = quote.close

        logger.info('---------------------------BUY-------------------------------------')
        logger.info(quote.symbol)
        logger.info(quote.date)
        logger.info(ref_price)
        logger.info(size)
        logger.info('-------------------------------------------------------------------\n')
        task_id, msg = self.ctx.trade_api.place_order(quote.symbol,
                                                      common.ORDER_ACTION.BUY,
                                                      ref_price,
                                                      self.buy_size_unit * size)

        if (task_id is None) or (task_id == 0):
            logger.info("place_order FAILED! msg = {}".format(msg))
        else:
            self.cur_holding_count += 1
    
    def sell(self, quote, size):
        if isinstance(quote, Quote):
            ref_price = (quote.bidprice1 + quote.askprice1) / 2.0
        else:
            ref_price = quote.close

        logger.info('---------------------------SELL-------------------------------------')
        logger.info(quote.symbol)
        logger.info(quote.date)
        logger.info(ref_price)
        logger.info(size)
        logger.info('--------------------------------------------------------------------\n')
        task_id, msg = self.ctx.trade_api.place_order(quote.symbol,
                                                      common.ORDER_ACTION.SELL,
                                                      ref_price,
                                                      self.buy_size_unit * size)

        if (task_id is None) or (task_id == 0):
            logger.info("place_order FAILED! msg = {}".format(msg))
        else:
            self.cur_holding_count -= 1
    
    """
    'on_tick' 接收单个quote变量，而'on_bar'接收多个quote组成的dictionary
    'on_tick' 是在tick级回测和实盘/仿真交易中使用，而'on_bar'是在bar回测中使用
    """
    def on_tick(self, quote):
        pass

    def on_bar(self, quote_dic):
        """
        这里传入的'quote'可以是:
            - Quote类型 (在实盘/仿真交易和tick级回测中，为tick数据)
            - Bar类型 (在bar回测中，为分钟或日数据)
        我们通过isinsance()函数判断quote是Quote类型还是Bar类型
        """
        self.quotelist = []
    
        for s in self.symbol:
            self.quotelist.append(quote_dic.get(s))
        for quote in self.quotelist:
            # print(quote)
            self.price_arr[quote.symbol] = self.price_arr[quote.symbol].append({'low': quote.low,
                                                                                'high': quote.high,
                                                                                'close': quote.close},
                                                                               ignore_index=True
                                                                               )
            # print(self.price_arr[quote.symbol])
            
        self.window_count += 1
        # print(self.window_count)
        if self.window_count <= self.window:
            return

        stockholdings = self.ctx.pm.holding_securities
        self.cur_holding_count = len(stockholdings)

        # 计算K,D,J,MACDhist
        for quote in self.quotelist:
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

            if df.iloc[-3,0] == 0 or df.iloc[-2,0] == 0 or df.iloc[-1,0] == 0:
                continue
            j = df['j']
            df['norm_j'] = j.apply(lambda x: (x - j.mean()) / (j.std()))
            bar_value = df['MACDhist']*2
            logger.info(quote.symbol)
            # print(df)
            df['norm_bar'] = bar_value.apply(lambda x: (x - bar_value.mean()) / (bar_value.std()))
            pic_path = self.create_macd_j_pic(quote.symbol, df[-3:], self.window_count)

            result = run_inference_on_image(pic_path)
            logger.info(result)

            # 交易逻辑：最大15支持仓股票，如果买入信号为买入并且还有剩余资金，则买入；持仓第5天平仓
            logger.info(self.holding_day[quote.symbol])
            if self.holding_day[quote.symbol] >= 4 and quote.symbol != self.benchmark_symbol:
                self.sell(quote, self.pos[quote.symbol])

            if quote.symbol != self.benchmark_symbol:
                if result == 1:
                    if self.pos[quote.symbol] == 0 and self.cur_holding_count < self.holding_count:
                        hands = np.floor(self.stock_value / quote.close / 100)
                        if self.balance >= self.stock_value and hands > 0:
                            self.buy(quote, hands*100)
                    else:
                        self.holding_day[quote.symbol] += 1
                else:
                    if self.pos[quote.symbol] > 0:
                        self.holding_day[quote.symbol] += 1


    def on_trade(self, ind):
        """
        交易完成后通过self.ctx.pm.get_pos得到最新仓位并更新self.pos
        """
        logger.info("\nStrategy on trade: ")
        logger.info(ind)

        for s in self.symbol:
            self.pos[s] = self.ctx.pm.get_pos(s)

        logger.info('\n---updated pos----\n')

        if common.ORDER_ACTION.is_positive(ind.entrust_action):
            self.balance -= ind.fill_price * ind.fill_size
            self.holding_day[ind.symbol] += 1
        else:
            self.balance += ind.fill_price * ind.fill_size
            self.holding_day[ind.symbol] = 0
        
        logger.info(ind.symbol)
        logger.info(self.balance)
        logger.info('\n')

    def on_order_status(self, ind):
        if self.output:
            logger.info("\nStrategy on order status: ")
            logger.info(ind)
            logger.info('\n')

    def on_task_status(self, ind):
        if self.output:
            logger.info("\nStrategy on task ind: ")
            logger.info(ind)
            logger.info('\n')


def run_strategy():
    if is_backtest:
        """
        回测模式
        """

        ds = RemoteDataService()
        ds.init_from_config(data_config)
        symbol_list = ds.query_index_member(index, start_date, start_date)
        # symbol_list = ['600887.SH']
        # symbol_list = sample(symbol_list, 20)
        print(symbol_list)

        # add the benchmark index to the last position of symbol_list
        symbol_list.append(index)
        props = {"symbol": ','.join(symbol_list),
                 "holding_Count": 15,
                 "start_date": start_date,
                 "end_date": end_date,
                 "bar_type": "1d",  # '1d'
                 "init_balance": 45000,
                 "commission_rate": 2E-4}

        tapi = BacktestTradeApi()
        ins = EventBacktestInstance()
        
    else:
        """
        实盘/仿真模式
        """
        props = {'symbol': '600519.SH',
                 "fast_ma_length": 5,
                 "slow_ma_length": 15,
                 'strategy.no': 1062}
        tapi = RealTimeTradeApi(trade_config)
        ins = EventLiveTradeInstance()

    props.update(data_config)
    props.update(trade_config)
    
    ds = RemoteDataService()
    strat = RNNStrategy()
    pm = PortfolioManager()
    
    context = model.Context(data_api=ds, trade_api=tapi, instance=ins,
                            strategy=strat, pm=pm)
    
    ins.init_from_config(props)
    if not is_backtest:
        ds.subscribe(props['symbol'])

    ins.run()
    if not is_backtest:
        time.sleep(9999)
    ins.save_results(folder_path=result_dir_path)
    
    ta = ana.EventAnalyzer()
    
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    ta.initialize(data_server_=ds, file_folder=result_dir_path)
    df_bench, _ = ds.daily(index, start_date=start_date, end_date=end_date)
    ta.data_benchmark = df_bench.set_index('trade_date').loc[:, ['close']]

    ta.do_analyze(result_dir=result_dir_path, selected_sec=props['symbol'].split(',')[:-1])


if __name__ == "__main__":
    run_strategy()