import multiprocessing
from jaqs.data import DataApi
import pandas as pd
import os
from random import sample
from shutil import copyfile


def create_macd_j_pic(symbol, df, sequence):
    import os
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    if not os.path.exists('./data/prepared/pic_data/macd_j_pic2/up'):
        os.makedirs('./data/prepared/pic_data/macd_j_pic2/up')
    if not os.path.exists('./data/prepared/pic_data/macd_j_pic2/down'):
        os.makedirs('./data/prepared/pic_data/macd_j_pic2/down')
    if not os.path.exists('./data/prepared/pic_data/macd_j_pic2/notup'):
        os.makedirs('./data/prepared/pic_data/macd_j_pic2/notup')

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
        fig.savefig('./data/prepared/pic_data/macd_j_pic2/up/' + str_symbol + '_' + str(sequence), dpi=20)
    elif df['close_five'][2] <= -0.05:
        fig.savefig('./data/prepared/pic_data/macd_j_pic2/down/' + str_symbol + '_' + str(sequence), dpi=20)
    else:
        fig.savefig('./data/prepared/pic_data/macd_j_pic2/notup/' + str_symbol + '_' + str(sequence), dpi=20)

    plt.close()

    print(sequence)


def choice_pics(class_path):
    class_path_up = class_path + 'up/'
    class_path_down = class_path + 'down/'
    class_path_notup = class_path + 'notup/'

    file_count = 0

    file_list_up = os.listdir(class_path_up)
    file_list_down = os.listdir(class_path_down)
    file_list_notup = os.listdir(class_path_notup)

    file_up_count = len(file_list_up)
    file_down_count = len(file_list_down)
    if file_down_count <= file_up_count:
        file_count = file_down_count
    else:
        file_count = file_up_count
    print(file_count)

    file_train_count = int(file_count*0.7)
    print(file_train_count)

    # 从目录中取一样多个图片
    file_list_up = sample(file_list_up, file_count)
    file_list_down = sample(file_list_down, file_count)
    file_list_notup = sample(file_list_notup, file_count)
    # 训练图片与测试图片7,3分
    file_list_up_train = sample(file_list_up,file_train_count)
    file_list_up_val = list(set(file_list_up) - set(file_list_up_train))
    file_list_down_train = sample(file_list_down, file_train_count)
    file_list_down_val = list(set(file_list_down) - set(file_list_down_train))
    file_list_notup_train = sample(file_list_notup,file_train_count)
    file_list_notup_val = list(set(file_list_notup) - set(file_list_notup_train))

    for i, img_name in enumerate(file_list_up_train):
        img_path = class_path_up + img_name
        copyfile(img_path, './data/train/up/' + img_name)
    for i, img_name in enumerate(file_list_down_train):
        img_path = class_path_down + img_name
        copyfile(img_path, './data/train/down/' + img_name)
    for i, img_name in enumerate(file_list_notup_train):
        img_path = class_path_notup + img_name
        copyfile(img_path, './data/train/notup/' + img_name)

    for i, img_name in enumerate(file_list_up_val):
        img_path = class_path_up + img_name
        copyfile(img_path, './data/validation/up/' + img_name)
    for i, img_name in enumerate(file_list_down_val):
        img_path = class_path_down + img_name
        copyfile(img_path, './data/validation/down/' + img_name)
    for i, img_name in enumerate(file_list_notup_val):
        img_path = class_path_notup + img_name
        copyfile(img_path, './data/validation/notup/' + img_name)


if __name__ == '__main__':
    # multiprocessing.freeze_support()
    # pool = multiprocessing.Pool()
    
    # api = DataApi(addr="tcp://data.quantos.org:8910")
    # result, msg = api.login("18652420434", "eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MTcwNjAxMDgyOTMiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg2NTI0MjA0MzQifQ.b1ejSpbEVS7LhbsveZ5kvbWgUs7fnUd0-CBakPwNUu4")
    # # print(result)
    # data, msg = api.query(
    #         view="lb.indexCons",
    #         fields="symbol",
    #         filter="index_code=000016.SH&start_date=20180801&end_date=20180831",
    #         data_format='pandas')
    # print(msg)
    # symbols = data['symbol'].tolist()
    # print(symbols)

    # for sym in symbols:
    #     df = pd.read_csv('./data/prepared/datacsv/' + sym + '.csv', sep=',')
    #     df.index = pd.to_datetime(df['trade_date'], format='%Y/%m/%d')
    #     df.reset_index(drop=True, inplace=True)
    #     j = df['j']
    #     df['norm_j'] = j.apply(lambda x: (x - j.mean()) / (j.std()))
    #     bar_value = df['MACDhist'] * 2
    #     df['norm_bar'] = bar_value.apply(lambda x: (x - bar_value.mean()) / (bar_value.std()))

    #     for i in range(len(df) - 8):
    #         ts = df[0 + i:3 + i]
    #         pool.apply(create_macd_j_pic, args=(sym, ts, i))
    choice_pics('./data/prepared/pic_data/macd_j_pic2/')
