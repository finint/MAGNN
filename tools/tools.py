import numpy as np
import pandas as pd
from datetime import datetime,timedelta
from itertools import chain
from operator import itemgetter
import os

# please change these relative path to path
parent_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# global constants
ORI_PRICE_DIR = os.path.join(parent_dir,'data/price')
EVENT_DIR = os.path.join(parent_dir,'data/event')
NEWS_DIR = os.path.join(parent_dir,'data/news')
PRICE_SEGMENT_DIR = os.path.join(parent_dir,'data/price-segment')
EVENT_EMBEDDING_SEGMENT_DIR = os.path.join(parent_dir,'data/event-embedding-segment')
NEWS_EMBEDDING_SEGMENT_DIR = os.path.join(parent_dir,'data/news-embedding-segment')
EVENT_NEWS_EMBEDDING_MERGE_DIR = os.path.join(parent_dir,'data/event-news-embedding-merge')
PRICE_EMBEDDING_MERGE_DIR = os.path.join(parent_dir,'data/price-embedding-merge')
PRICE_NUMPY_DIR = os.path.join(parent_dir,'data/price-numpy')
PRICE_EMBEDDING_DIR = os.path.join(parent_dir,'data/price-embedding')
EVENT_NEWS_RESULT = os.path.join(parent_dir,'data/event-news-result')
PRICE_GAT_RESULT = os.path.join(parent_dir,'data/price-gat-result')

ONE_DAY_HIGH_RETURN = 'one_day_high_return'
ONE_DAY_OPEN_RETURN = 'one_day_open_return'
ONE_DAY_LOW_RETURN = 'one_day_low_return'
ONE_DAY_CLOSE_RETURN = 'one_day_close_return'
NEXT_DAY_CLOSE_RETURN = 'next_day_close_return'
LABEL = 'label'
LABEL_NORMAL = 'label_normal'

RISING_LABLE = 0
FALLING_LABLE = 1
MEDIUM_LABLE = 2

# 阈值设置，需要根据收益分布考虑调整
MEDIUM_RANGE = (-0.01, 0.01)

# some functions in this file DO NOT return reliable values, they're only used for demonstration

# reader might use packages to update these functions(pandas-market-calendars, for instence)
def get_trading_dates(start_dt,end_dt):
    dates=[]
    stdt=pd.to_datetime(start_dt)
    endt=pd.to_datetime(end_dt)
    cnt=stdt
    while cnt<=endt:
        dates.append(cnt)
        cnt+=timedelta(1)
    return dates

# this function returns NATURE DATES instead of trading dates
def get_previous_trading_date(cur,days=1):
    current=pd.to_datetime(cur)
    previous=current - timedelta(days)
    return previous

# this function returns NATURE DATES instead of trading dates
def get_next_trading_date(cur,days=1):
    current=pd.to_datetime(cur)
    next=current+timedelta(days)
    return next
    
# get price dataframe from database
def get_price(kdcodes, start_date, end_date, fields):
    result = pd.DataFrame()
    cur = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    while cur <= end:
        dt = datetime.strptime(str(cur), '%Y-%m-%d %H:%M:%S')
        dt = dt.strftime('%Y-%m-%d')
        path = os.path.join(ORI_PRICE_DIR, dt + '.pkl')
        temp = pd.read_pickle(path)
        temp=temp[temp['kdcode'].isin(kdcodes)]
        result = pd.concat([result, temp])  # Store the merged DataFrame in result
        cur += timedelta(1)

    return result

# get stocklist from selected date price dataframe
def get_cur_stocklist(dt):
    dt=pd.to_datetime(dt)
    strdt=datetime.strftime(dt,'%Y-%m-%d')
    filepath=strdt+'.pkl'
    df=np.load(os.path.join(ORI_PRICE_DIR,filepath))
    stocklist=df['kdcode'].unique()    
    return stocklist    

# clean ' | ' in column:'kdcodes'
def chainer(s):
    return list(chain.from_iterable(s.str.split('|')))

def remove_nan(x):
    if len(x.price_embedding_list) <= 1:
        return x
    not_nan_index = np.where(pd.isna(x.price_embedding_list) == False)[0]
    x.price_embedding_list = list(itemgetter(*not_nan_index)(x.price_embedding_list))
    x.price_kdcode_list = list(itemgetter(*not_nan_index)(x.price_kdcode_list))
    return x


    