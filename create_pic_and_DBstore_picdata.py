import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import talib
import pymysql
import cv2
import json
import time,logging
import sys
import os


logger = logging.getLogger('prepare_tf_data')
logger.setLevel(logging.DEBUG)
# 创建一个handler，用于写入日志文件
fh = logging.FileHandler('/prepare_tf_data.log',encoding='UTF-8')
fh.setLevel(logging.DEBUG)
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)



def connect_db():
    try:
        db = pymysql.connect("45.76.77.76", "root", "123456654321", "stockMarketMoodAnalysis", charset='utf8')
        cursor = db.cursor()
        return {"db": db, "cursor": cursor,"result":1}
    except:
        return {"db": '', "cursor": '',"result":0}



def modify_data(db,cursor,modify_sql):
    try:
        cursor.execute(modify_sql)
        db.commit()
        return 1
    except():
        db.rollback()
        print("insert_sql: %s"%modify_sql)
        logger.info("insert_sql: %s"%modify_sql)
        logger.info("modify_db_fail")
        return 0
def db_close(db):
    db.close()


#获取数据
def get_tushare_data(symbol):

    # df=ts.get_hist_data(symbol,start='2016-01-01',end='2018-04-30')#tushare拿数据
    df = ts.get_k_data(symbol, start='2008-05-01', end='2018-04-30')
    # df = df.set_index("date")
    close = [float(x) for x in df['close']]
    try:
        # 调用talib计算MACD指标,返回的MACD是DIF，快线；MACDsignal是DEA值，慢线；macd=dif-DEA画成柱状图
        df['MACD'], df['MACDsignal'], df['MACDhist'] = talib.MACD(np.array(close),
                                                                  fastperiod=12, slowperiod=26, signalperiod=9)
        df = df.sort_index()
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
        df = lm_kdj(df, 9, 'close')  ##计算好kdj之后从13行开始取数据,计算出来的kdj比较准确
        df = df[34:]
        df.to_csv(path_or_buf='F:\Code\\buysell\data\pic_data\datacsv\\'+symbol+'.csv',sep=',',index=True)
    except:
        # print(symbol)
        # print(df)
        print("Unexpected error:", sys.exc_info()[0])

def deal_close_datas(symbol,df,sequence):
    df = df.set_index("date")
    df = df.sort_index()
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d') ##设置时间为index之后再画图，不设置话kdj效果更好
    close = df.iloc[[sequence + 29]].close.values[0]
    start_time=df.iloc[[sequence]].close
    start_time = dict(start_time)
    for key in start_time:
        start_time=key
    end_time = df.iloc[[sequence+29]].close
    end_time = dict(end_time)
    for key in end_time:
        end_time = key
    # print(df.iloc[[i + 29]].close)
    comp_days_close=["day_one","day_two","day_three","day_four","day_five"]
    days_close=["day_one_close","day_two_close","day_three_close",
                "day_four_close","day_five_close"]
    aa=[]
    for j in range(len(comp_days_close)):
        # print(df.iloc[[i + 30 + j]].close)
        days_close[j] = df.iloc[[sequence + 30 + j]].close.values[0]
        # print(days_close[j])
        if (days_close[j] - close)/close > 0.05:
            comp_days_close[j] = 1
        elif (days_close[j] - close)/close < -0.05:
            comp_days_close[j] = -1
        else:
            comp_days_close[j] = 0

    data_dict = {"stock": symbol, "sequence": sequence, "close": close,
                 "day_one": comp_days_close[0], "day_two": comp_days_close[1], "day_three": comp_days_close[2],
                 "day_four": comp_days_close[3], "day_five": comp_days_close[4],
                 "day_one_close": days_close[0], "day_two_close": days_close[1], "day_three_close": days_close[2],
                 "day_four_close": days_close[3], "day_five_close": days_close[4],"start_time":start_time,"end_time":end_time}

    return data_dict




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

def create_kdj_pic(symbol,df,sequence):
    # df = df.set_index("date")
    # df = df.sort_index()
    # df.index = pd.to_datetime(df.index, format='%Y-%m-%d') ##设置时间为index之后再画图，不设置话kdj效果更好
    if not os.path.exists('F:\Code\\buysell\data\pic_data\kdj_pic\\' + symbol):
        os.makedirs('F:\Code\\buysell\data\pic_data\kdj_pic\\' + symbol)
    fig=plt.gcf()
    # fig = plt.figure(figsize=(12.8,12.8))  ###设置图形的大小  figsize=(12.8,12.8) 保存的时候dpi=10可以得到128*128的图片
    print(df)
    try:
        sig_k=df.k
        sig_d=df.d
        sig_j=df.k*3-df.d*2
        plt.plot(sig_k.index,sig_k,label='k')
        plt.plot(sig_d.index,sig_d,label='d')
        plt.plot(sig_j.index,sig_j, label='j')
        plt.axis('off')
        # plt.show()
        fig.savefig('F:\Code\\buysell\data\pic_data\kdj_pic\\'+symbol+'\\'+str(sequence))
        plt.close()
        return 1
    except:
        logger.info('stock:%s;kdj画图到第%s失败'%(symbol,str(sequence)))
        return 0
# df = pd.read_csv('E:\pic_data\datacsv\\' + '600000' + '.csv', sep=',')
# create_kdj_pic("600000",df[0:30],0)


def create_macd_pic(symbol,df,sequence):
    if not os.path.exists('F:\Code\\buysell\data\pic_data\macd_pic\\' + symbol):
        os.makedirs('F:\Code\\buysell\data\pic_data\macd_pic\\' + symbol)
    fig = plt.gcf()
    try:
        plt.plot(df.index, df['MACD'], label='macd dif')#快线
        plt.plot(df.index, df['MACDsignal'], label='signal dea')# 慢线
        plt.bar(df.index, df['MACDhist']*2, label='hist bar')
        plt.axis('off')
        fig.savefig('F:\Code\\buysell\data\pic_data\macd_pic\\' + symbol + '\\' + str(sequence))
        plt.close()
        return 1
    except:
        logger.info('stock:%s;macd画图到第%s失败' % (symbol, str(sequence)))
        return 0

def create_pics_and_datas(symbol,pic_type=None):
    data_dict_list = []
    df = pd.read_csv('F:\Code\\buysell\data\pic_data\datacsv\\' + symbol + '.csv', sep=',')
    ##计算kdj
    # df=lm_kdj(df,9,'close')  ##计算好kdj之后从13行开始取数据
    # df=df[13:]
    pic_count=0
    for i in range(len(df)-34):
        if pic_type=="kdj":
            pic_res=create_kdj_pic(symbol, df[0 + i:30 + i], i)  # 画图的
            if pic_res==1:
                pic_count=pic_count+1
        elif pic_type=="macd":
            pic_res=create_macd_pic(symbol, df[0 + i:30 + i], i)  # 画图的
            if pic_res==1:
                pic_count=pic_count+1
        close_data = deal_close_datas(symbol,df,i)

        data_dict_list.append(close_data)
    print("stock:"+symbol+';画图数量：'+str(pic_count))
    logger.info('stock:%s;画图数量%s' % (symbol, str(pic_count)))
    return data_dict_list

def store_csv(symbol,data_dict_list,pic_type=None):
    df = pd.DataFrame(data_dict_list)
    # print(df)
    df.to_csv(path_or_buf='F:\Code\\buysell\data\pic_data\datacsv\\'+symbol+'_final.csv',sep=',',index=True)


def store_database(symbol,data_dict_list,db_resp,pic_type=None):

    db = db_resp['db']
    cursor = db_resp['cursor']
    data_count = 0
    if pic_type == "kdj":
        path = 'E:\pic_data\kdj_pic\\'
    else:
        path = 'E:\pic_data\macd_pic\\'
    for i in range(len(data_dict_list)):
        data_dict=data_dict_list[i]
        img_read = cv2.imread(path+symbol+'\\'+str(i)+'.png', 2)
        img = cv2.resize(img_read, (128, 128))

        img_list = img.tolist()
        img_dict = {}
        img_dict['name'] = str(i)
        img_dict['content'] = img_list
        img_json_data = json.dumps(img_dict)

        ##这段代码验证放到数据库里的矩阵能否画图
        # img_json_data = json.loads(img_json_data)
        # img_list = img_json_data['content']
        # img = np.asarray(img_list)
        # # np.savetxt('E:\\new.csv', img, delimiter=',')
        # img=img.astype(dtype="uint8")
        # cv2.namedWindow("Image")
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #################
        create_dat=str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        insert_sql = "INSERT INTO image_array_data_test (stock,sequence,close_data,day_one,day_two,day_three,day_four,day_five,day_one_close,day_two_close,day_three_close,day_four_close,day_five_close,start_time,end_time,image_array,create_dat) VALUES ('%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s')"%(data_dict["stock"],data_dict["sequence"],data_dict["close"],
                        data_dict["day_one"],data_dict["day_two"],data_dict["day_three"],
                        data_dict["day_four"],data_dict["day_five"],
                        data_dict["day_one_close"],data_dict["day_two_close"],
                        data_dict["day_three_close"],data_dict["day_four_close"],data_dict["day_five_close"],
                        data_dict["start_time"],data_dict["end_time"],pymysql.escape_string(img_json_data),create_dat)
        # s_time=time.time()
        modify_data(db, cursor, insert_sql)
        data_count=data_count+1
        # e_time=time.time()
        # legency=e_time-s_time
    print("stock:"+symbol+'存入数据条数：'+str(data_count))
    logger.info('stock:%s;存入数据条数%s' % (symbol, str(data_count)))


##画图并保存图片到本地。从本地读图片数据并保存到远程数据库
stocks_list=['601328','600999', '601628', '600016', '601985', '600019', '600028', '601668', '601878', '601688', '601669', '601390', '601211', '601006', '601989', '601988', '600104', '600518', '601398', '600958', '600309', '601857','600030', '601318', '600837', '600036', '600050', '600547', '601766', '601166', '601601', '601229', '600919', '601169', '600029', '601800', '600000', '600887', '600340', '601881', '601088','603993', '600519', '601336', '601186', '600606', '601818', '600048', '600111', '601288' ]
# stocks_list=['600000']
# get_tushare_data(stocks_list[0])
# for stock in stocks_list:
#     get_tushare_data(stock)
dat = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
logger.info('start_time:%s'%dat)
# db_resp = connect_db()
i=1
for stock in stocks_list:
    print(stock)
    start_time=time.time()
    data_dict_list=create_pics_and_datas(stock,pic_type='macd')
    end_time=time.time()
    logger.info('stock:%s;画图耗时%ss' % (stock, str(end_time-start_time)))
    store_start_time=time.time()
    # store_database(stock,data_dict_list,db_resp,pic_type='macd')
    store_csv(stock,data_dict_list,pic_type='macd')
    store_end_time=time.time()
    store_latency=str(store_end_time-store_start_time)
    stock_end_time = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    logger.info('stock:%s;存入CSV耗时%ss,结束时间：%s' % (stock, store_latency,stock_end_time))
    logger.info('进度:%d' % i)
    i=i+1

# db_close(db_resp['db'])
end_dat = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
logger.info('end_time:%s' %str(end_dat))




