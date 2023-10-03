
import math
import os
from bert_serving.client import BertClient
import numpy as np

import sys
sys.path.append('.')

from prepossessing.company_relation import get_company_relation

from operator import itemgetter

from datetime import datetime

import time

RISING_LABLE = 0
FALLING_LABLE = 1
MEDIUM_LABLE = 2

FEATURE_NUM = 4
LABEL_NUM = 3


from tools.tools import *

company_relation = get_company_relation()
company_relation = company_relation.dropna()

# 阈值设置，需要根据收益分布考虑调整
MEDIUM_RANGE = (-0.01, 0.01)


""" PART1 : FOR GENERATING PRICE LABELS """
# get daily return through price_df
def get_one_day_return(close_price_ser, periods=1):
    shift_close_price_ser = close_price_ser.shift(periods=periods)
    if periods == 1:
        one_day_return = (close_price_ser - shift_close_price_ser) / shift_close_price_ser
    else:
        one_day_return = (shift_close_price_ser - close_price_ser) / close_price_ser
    return one_day_return

# get price labels
def get_label_from_close_return(one_day_return):
    if math.isnan(one_day_return):
        return None
    else:
        if one_day_return > MEDIUM_RANGE[1]:
            return RISING_LABLE
        elif one_day_return < MEDIUM_RANGE[0]:
            return FALLING_LABLE
        else:
            return MEDIUM_LABLE


# 剔除当日停牌，st股票，或者新上市未满1年股票 
def not_special(kdcode, dt, list_year):
    return kdcode


def get_price_feature_and_label(dt):
    dt = str(dt)
    dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
    dt = dt.strftime('%Y-%m-%d')
        
    file_name = dt + '.pkl'
    if os.path.exists(os.path.join(ORI_PRICE_DIR, file_name)):
        print('price path exists!')
        price_df = pd.read_pickle(os.path.join(ORI_PRICE_DIR, file_name))
        return price_df
    
    
    stock_list=get_cur_stocklist(pd.to_datetime(dt))
    
    # remove special stocks
    stock_list = not_special(stock_list, dt, 1)

    price_df = get_price(kdcodes=stock_list, start_date=get_previous_trading_date(dt),
                         end_date=get_next_trading_date(dt),
                         fields=['high', 'open', 'low', 'close']).reset_index()

    # dealing nan vals
    price_df = price_df.dropna()

    price_df[ONE_DAY_HIGH_RETURN] = price_df['high'].groupby(price_df['kdcode']).transform(
        lambda x: get_one_day_return(x))
    price_df[ONE_DAY_OPEN_RETURN] = price_df['open'].groupby(price_df['kdcode']).transform(
        lambda x: get_one_day_return(x))
    price_df[ONE_DAY_LOW_RETURN] = price_df['low'].groupby(price_df['kdcode']).transform(
        lambda x: get_one_day_return(x))
    price_df[ONE_DAY_CLOSE_RETURN] = price_df['close'].groupby(price_df['kdcode']).transform(
        lambda x: get_one_day_return(x))
    price_df[NEXT_DAY_CLOSE_RETURN] = price_df['close'].groupby(price_df['kdcode']).transform(
        lambda x: get_one_day_return(x, periods=-1))
    price_df[LABEL] = price_df[NEXT_DAY_CLOSE_RETURN].apply(lambda x: get_label_from_close_return(x))
    
    price_df.to_csv('pricedf.csv')
    
    price_df[LABEL_NORMAL] = (price_df[NEXT_DAY_CLOSE_RETURN] - (-0.1)) / (0.1 - (-0.1))
    price_with_label_df = price_df[price_df['dt'] == pd.Timestamp(dt)]
    pd.to_pickle(price_with_label_df, os.path.join(ORI_PRICE_DIR, file_name))
    return price_with_label_df

from embedding_as_service.text.encode import Encoder
en = Encoder(embedding='bert', model='bert_base_cased', max_seq_length=256)


# use bert to do embedding for event&news content
def get_embedding_from_bert(df):
    df = df.dropna()
    # 因为encode为3维向量所以求mean
    df['embedding'] = np.mean(en.encode(list(df['content'])), axis=1).tolist()
    df.to_csv('embedding.csv')
    return df

# get event&news embedding dataframes
def get_event_news_embedding_df(dt, type):
    dt=str(dt)
    dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
    dt = dt.strftime('%Y-%m-%d')
    
    if type == 'event':
        file_path = os.path.join(EVENT_DIR, dt + '.pkl')
        embedding_file_path = os.path.join(EVENT_DIR, dt + '-embedding.pkl')
    elif type == 'news':
        file_path = os.path.join(NEWS_DIR, dt + '.pkl')
        embedding_file_path = os.path.join(NEWS_DIR, dt + '-embedding.pkl')

    if type == 'event':
        if os.path.exists(embedding_file_path):
            embedding_df = pd.read_pickle(embedding_file_path)
        else:
            if not os.path.exists(file_path):
                print(file_path,'Please get your own data from your database')
                return
            else:
                print('load success!')
                df = pd.read_pickle(file_path)
            if len(df) != 0:
                embedding_df = get_embedding_from_bert(df)
            else:
                embedding_df = pd.DataFrame()
            embedding_df.to_pickle(embedding_file_path)
    elif type == 'news':
        if os.path.exists(embedding_file_path):
            embedding_df = pd.read_pickle(embedding_file_path)
        else:
            if not os.path.exists(file_path):                
                print(file_path,'Please get your own data from your database')
            else:
                print('load success!')
                df = pd.read_pickle(file_path)
            if len(df) != 0:
                embedding_df = get_embedding_from_bert(df)
            else:
                embedding_df = pd.DataFrame()
            embedding_df.to_pickle(embedding_file_path)

    return embedding_df


# generate price segment data for each tuple
def get_price_segment(start_dt, end_dt):
    file_name = "-".join([start_dt, end_dt]) + '.pkl'
    file_path = os.path.join(PRICE_SEGMENT_DIR, file_name)
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    
    dts = get_trading_dates(start_dt, end_dt)
    price_dfs = []
    for dt in dts:
        price_df = get_price_feature_and_label(dt)
        price_dfs.append(price_df)

    result = pd.concat(price_dfs)
    pd.to_pickle(result, file_path)
    return result

# generate event&news segment data for each tuple
def get_event_news_embedding_segment(start_dt, end_dt, type):
    file_name = "-".join([start_dt, end_dt]) + '.pkl'

    if type == 'event':
        file_path = os.path.join(EVENT_EMBEDDING_SEGMENT_DIR, file_name)
    elif type == 'news':
        file_path = os.path.join(NEWS_EMBEDDING_SEGMENT_DIR, file_name)
    
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)

    dts = get_trading_dates(start_dt, end_dt)
    embedding_dfs = []
    for dt in dts:
        embedding_df = get_event_news_embedding_df(dt, type)
        embedding_dfs.append(embedding_df)

    result = pd.concat(embedding_dfs)
    pd.to_pickle(result, file_path)
    return result

# merge event&news data by date_tuples
def get_event_news_embedding_merge_data(date_tuples):
    event_news_embedding_merge_dfs = []
    for date_tuple in date_tuples:
        df = get_event_news_embedding_merge_data_by_date_range(date_tuple[0], date_tuple[1])
        event_news_embedding_merge_dfs.append(df)
    result = pd.concat(event_news_embedding_merge_dfs)
    return result


# merge event&news embeddings intto price dataframe
def get_event_news_embedding_merge(kdcode, dt,event_df, news_df):        
    index=(kdcode,dt)
    try:
        rel_event = event_df.loc[index]
    except Exception as e:
        rel_event = []
            
    if len(rel_event) == 0:
        rel_event_list = []
        rel_event_id_list = []
    else:
        rel_event_list = rel_event['embedding'].to_list()
        rel_event_id_list = rel_event['id'].to_list()

    try:
        rel_news = news_df.loc[index]
    except Exception as e:
        rel_news = []

    if len(rel_news) == 0:
        rel_news_list = []
        rel_news_id_list = []
    else:
        rel_news_list = rel_news['embedding'].to_list()
        rel_news_id_list = rel_news['id'].to_list()
    
    return [rel_event_list, rel_event_id_list, rel_news_list, rel_news_id_list]

# merge event&news embedding data by date range
def get_event_news_embedding_merge_data_by_date_range(start_dt, end_dt):
    file_name = "-".join([start_dt, end_dt]) + '.pkl'
    
    file_path = os.path.join(EVENT_NEWS_EMBEDDING_MERGE_DIR, file_name)

    if os.path.exists(file_path):
        return pd.read_pickle(file_path)

    price_df = get_price_segment(start_dt, end_dt)

    event_embedding_df = get_event_news_embedding_segment(start_dt, end_dt, 'event')
    news_embedding_df = get_event_news_embedding_segment(start_dt, end_dt, 'news')
    
    lens = event_embedding_df['kdcodes'].str.split('|').map(len)
   
    # create new dataframe, repeating or chaining as appropriate
    event_embedding_df = pd.DataFrame({'id': np.repeat(event_embedding_df['id'], lens),
                                       'dt': np.repeat(event_embedding_df['dt'], lens),
                                       'content': np.repeat(event_embedding_df['content'], lens),
                                       'embedding': np.repeat(event_embedding_df['embedding'], lens),
                                       'kdcodes': chainer(event_embedding_df['kdcodes'])})
    event_embedding_df['kdcode'] = event_embedding_df['kdcodes']
    
    lens = news_embedding_df['kdcodes'].str.split('|').map(len)

    news_embedding_df = pd.DataFrame({'id': np.repeat(news_embedding_df['id'], lens),
                                      'dt': np.repeat(news_embedding_df['dt'], lens),
                                      'content': np.repeat(news_embedding_df['content'], lens),
                                      'embedding': np.repeat(news_embedding_df['embedding'], lens),
                                      'kdcodes': chainer(news_embedding_df['kdcodes'])})
    news_embedding_df['dt'] = pd.to_datetime(news_embedding_df['dt'])
    news_embedding_df['kdcode'] = news_embedding_df['kdcodes']
    
        
    price_df[['event_embedding_list', 'event_id_list', 'news_embedding_list', 'news_id_list']] = \
        price_df.apply(lambda x: pd.Series(get_event_news_embedding_merge(x['kdcode'],
                                                                          x['dt'],
                                                                          event_embedding_df.set_index(
                                                                              ['kdcode', 'dt']),
                                                                          news_embedding_df.set_index(
                                                                              ['kdcode', 'dt']))),
                       axis=1)
    
    pd.to_pickle(price_df, file_path)
    return price_df

# correspond price with event&news through company relations
def get_price_embedding_merge(kdcode, dt, price_df, self_price_embedding):        
    relation_kdcodes = company_relation[company_relation['kdcode'] == kdcode]['relation_kdcode']
    try:
        if len(relation_kdcodes) != 0:
            relation_kdcodes = relation_kdcodes.iloc[0].split('|')

            index = []
            for relation_kdcode in relation_kdcodes:
                index.append((relation_kdcode, dt))

            rel_price_df = price_df.loc[index]
            rel_price = rel_price_df['price_embedding'].to_list()
            rel_kdcode_list = relation_kdcodes
        else:
            rel_price = []
            rel_kdcode_list = []
    except Exception as e:
        print(e)
    rel_price.append(self_price_embedding)
    rel_kdcode_list.append(kdcode)
    
    return [rel_price, rel_kdcode_list]

# load existed price embedding results
def get_price_embedding_df(period_tuple):
    dts = np.load(os.path.join(PRICE_NUMPY_DIR, '-'.join(period_tuple) + '-dts-test.npy'), allow_pickle=True)
    kdcodes = np.load(os.path.join(PRICE_NUMPY_DIR, '-'.join(period_tuple) + '-kdcodes-test.npy'), allow_pickle=True)
    labels = np.load(os.path.join(PRICE_NUMPY_DIR, '-'.join(period_tuple) + '-labels-test.npy'), allow_pickle=True)
    df = pd.DataFrame({'dt': dts, 'kdcode': kdcodes, 'label': labels})
    pred_label = np.load(
        os.path.join(PRICE_EMBEDDING_DIR, '-'.join(period_tuple) + '-lstm_model_label-test.npy'), allow_pickle=True)
    pred_probability = np.load(
        os.path.join(PRICE_EMBEDDING_DIR, '-'.join(period_tuple) + '-lstm_model_probability-test.npy'), allow_pickle=True)
    price_embedding = np.load(
        os.path.join(PRICE_EMBEDDING_DIR, '-'.join(period_tuple) + '-lstm_model_embedding-test.npy'), allow_pickle=True)
    df['price_embedding'] = price_embedding.tolist()
    df['pred_label'] = pred_label
    df['pred_probability'] = pred_probability.tolist()
    return df

#get magnn result
def get_magnn_result(period_tuple):
    result_path = r'./dataset/magnn-result/'
    model_name = 'magnn_model'
    lr = 0.005
    PRICE_EMBEDDING_LENGTH = 64
    MIDDLE_ALPHA_LENGTH = 64
    FINAL_WEIGHT_LENGTH = 64
    FINAL_ALPHA_LENGTH = 64

    ori_data = get_all_embedding_merge_data([period_tuple])

    label = np.load(os.path.join(result_path, '-'.join(
        period_tuple) + '-magnn_label-test-{lr}-{PRICE_EMBEDDING_LENGTH}-{MIDDLE_ALPHA_LENGTH}-{FINAL_WEIGHT_LENGTH}-{FINAL_ALPHA_LENGTH}.npy'.format(
        model_name=model_name,
        lr=lr,
        PRICE_EMBEDDING_LENGTH=PRICE_EMBEDDING_LENGTH,
        MIDDLE_ALPHA_LENGTH=MIDDLE_ALPHA_LENGTH,
        FINAL_WEIGHT_LENGTH=FINAL_WEIGHT_LENGTH,
        FINAL_ALPHA_LENGTH=FINAL_ALPHA_LENGTH)), allow_pickle=True)
    probability_np = np.load(os.path.join(result_path, '-'.join(
        period_tuple) + '-magnn_probability-test-{lr}-{PRICE_EMBEDDING_LENGTH}-{MIDDLE_ALPHA_LENGTH}-{FINAL_WEIGHT_LENGTH}-{FINAL_ALPHA_LENGTH}.npy'.format(
        model_name=model_name,
        lr=lr,
        PRICE_EMBEDDING_LENGTH=PRICE_EMBEDDING_LENGTH,
        MIDDLE_ALPHA_LENGTH=MIDDLE_ALPHA_LENGTH,
        FINAL_WEIGHT_LENGTH=FINAL_WEIGHT_LENGTH,
        FINAL_ALPHA_LENGTH=FINAL_ALPHA_LENGTH
    )), allow_pickle=True)
    event_alpha = np.load(os.path.join(result_path, '-'.join(
        period_tuple) + '-magnn_event_alpha-test-{lr}-{PRICE_EMBEDDING_LENGTH}-{MIDDLE_ALPHA_LENGTH}-{FINAL_WEIGHT_LENGTH}-{FINAL_ALPHA_LENGTH}.npy'.format(
        model_name=model_name,
        lr=lr,
        PRICE_EMBEDDING_LENGTH=PRICE_EMBEDDING_LENGTH,
        MIDDLE_ALPHA_LENGTH=MIDDLE_ALPHA_LENGTH,
        FINAL_WEIGHT_LENGTH=FINAL_WEIGHT_LENGTH,
        FINAL_ALPHA_LENGTH=FINAL_ALPHA_LENGTH)), allow_pickle=True)
    news_alpha = np.load(os.path.join(result_path, '-'.join(
        period_tuple) + '-magnn_news_alpha-test-{lr}-{PRICE_EMBEDDING_LENGTH}-{MIDDLE_ALPHA_LENGTH}-{FINAL_WEIGHT_LENGTH}-{FINAL_ALPHA_LENGTH}.npy'.format(
        model_name=model_name,
        lr=lr,
        PRICE_EMBEDDING_LENGTH=PRICE_EMBEDDING_LENGTH,
        MIDDLE_ALPHA_LENGTH=MIDDLE_ALPHA_LENGTH,
        FINAL_WEIGHT_LENGTH=FINAL_WEIGHT_LENGTH,
        FINAL_ALPHA_LENGTH=FINAL_ALPHA_LENGTH)), allow_pickle=True)
    final_alpha = np.load(os.path.join(result_path, '-'.join(
        period_tuple) + '-magnn_final_alpha-test-{lr}-{PRICE_EMBEDDING_LENGTH}-{MIDDLE_ALPHA_LENGTH}-{FINAL_WEIGHT_LENGTH}-{FINAL_ALPHA_LENGTH}.npy'.format(
        model_name=model_name,
        lr=lr,
        PRICE_EMBEDDING_LENGTH=PRICE_EMBEDDING_LENGTH,
        MIDDLE_ALPHA_LENGTH=MIDDLE_ALPHA_LENGTH,
        FINAL_WEIGHT_LENGTH=FINAL_WEIGHT_LENGTH,
        FINAL_ALPHA_LENGTH=FINAL_ALPHA_LENGTH)), allow_pickle=True)
    ori_data.rename(columns={'pred_label':'ori_pred_label'},inplace=True)
    final_alpha_df = pd.DataFrame(final_alpha, columns=['event_att', 'news_att', 'price_att'])
    probability = pd.DataFrame(probability_np, columns=['up', 'down', 'neural'])
    pred_label = pd.DataFrame(label, columns=['pred_label'])
    result = pd.concat([ori_data, final_alpha_df, probability, pred_label], axis=1)
    result['pred_probability'] = probability_np.tolist()
    return result[['kdcode', 'dt', 'label', 'event_att', 'news_att', 'price_att', 'up', 'down', 'neural', 'pred_label','ori_pred_label',
                   'pred_probability']]

# get price embedding merged data by date range
def get_price_embedding_merge_data_by_date_range(start_dt, end_dt):
    file_name = "-".join([start_dt, end_dt]) + '.pkl'

    file_path = os.path.join(PRICE_EMBEDDING_MERGE_DIR, file_name)

    if os.path.exists(file_path):
        return pd.read_pickle(file_path)

    price_embedding_df = get_price_embedding_df([start_dt, end_dt])

    price_embedding_df_index = price_embedding_df.set_index(['kdcode', 'dt'])

    price_embedding_df[['price_embedding_list', 'price_kdcode_list']] = \
        price_embedding_df.apply(lambda x: pd.Series(get_price_embedding_merge(x['kdcode'],
                                                                               x['dt'],
                                                                               price_embedding_df_index,
                                                                               x['price_embedding'])),
                                 axis=1)

    pd.to_pickle(price_embedding_df, file_path)
    return price_embedding_df

# get price embedding data which generated by lstm in price_embedding_lstm_.py
def get_price_embedding_merge_data(date_tuples):
    price_embedding_merge_dfs = []
    for date_tuple in date_tuples:
        df = get_price_embedding_merge_data_by_date_range(date_tuple[0], date_tuple[1])
        price_embedding_merge_dfs.append(df)
    result = pd.concat(price_embedding_merge_dfs)
    return result


# prepare selected time period data embeddings for magnn
def init_event_news_embeeding_df(start_dt, end_dt, type):
    dts = get_trading_dates(start_dt, end_dt)
    for dt in dts:
        embedding_df = get_event_news_embedding_df(dt, type)

# get all traning&testing data for magnn model
def get_all_embedding_merge_data(date_tuples):
    event_news_df = get_event_news_embedding_merge_data(date_tuples)
    event_news_df.drop(columns=['label'], inplace=True)
    price_df = get_price_embedding_merge_data(date_tuples)
    price_df_new = price_df.apply(lambda x: remove_nan(x), axis=1)
    # label在这里变成了price df的label，是直接从lstm 的result里面读出来的
    
    price_df_new['dt'] = pd.to_datetime(price_df_new['dt'])
    result = pd.merge(price_df_new, event_news_df, on=['kdcode', 'dt'])
    result['event_embedding_len'] = result['event_embedding_list'].apply(lambda x: len(x))
    result['news_embedding_len'] = result['news_embedding_list'].apply(lambda x: len(x))
    
    
    return result

# get all result for maggn
def get_all_result(date_tuples):
    result_dfs = []
    for date_tuple in date_tuples:
        df = get_magnn_result(date_tuple)
        result_dfs.append(df)

    result_df = pd.concat(result_dfs)

    if 'close' in result_df.columns:
        result_df.drop(columns=['close'], inplace=True)
    result_df['dt'] = result_df['dt'].astype(str)
    return result_df

    
