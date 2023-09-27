import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append('.')
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from prepossessing.constant import TEST_QUARTERS,ALL_QUARTER

from model.price_lstm_embedding import get_price_embedding_df
from prepossessing.event_news_embedding import get_event_news_embedding_df
from prepossessing.company_relation import get_company_relation
from prepossessing.magnn_dataset import MAGNN_Dataset
from prepossessing.data_prepare import get_all_embedding_merge_data

import time

from tools.tools import *

""" global defines"""
EVENT_NEWS_EMBEDDING_LENGTH = 768
#
lr = 0.005
PRICE_EMBEDDING_LENGTH = 64
MIDDLE_ALPHA_LENGTH = 64
FINAL_WEIGHT_LENGTH = 64
FINAL_ALPHA_LENGTH = 64
NUM_CLASSES = 3

stock_list=get_cur_stocklist('2019-04-01')#example

company_relation = get_company_relation()
company_relation = company_relation.dropna()

MAGNN_RESULT = './data/magnn-result'
MAGNN_LOG_PATH = './magnn/data/magnn-log'

"""get relationship embeddings by relation, which seems not used in magnn process
        Args:
            
            kdcode: specific kdcode for finding relationships
            dt: datetime 
            event_df: embedding dataframe for events
            news_df: embedding dataframe for news
            price_df: embedding dafaframe for price
            self_price_embedding: the stock's own price embedding            

        Returns:
            [rel_event_list, rel_event_id_list, rel_news_list, rel_news_id_list, rel_price]
            
            rel_event_list: related event embedding list
            rel_event_id_list: related event id list
            rel_news_list: related news embedding list
            rel_news_id_list: related news id list
            rel_price: the stock's own price on that specific date
            
            
        """
def get_relation_embedding_by_relation(kdcode, dt, event_df, news_df, price_df, self_price_embedding):
    dt_prev_20 = get_previous_trading_date(dt, 20)
    dt_prev_3 = get_previous_trading_date(dt, 3)
    rel_event = event_df[
        event_df['kdcodes'].str.contains(kdcode) & (event_df['dt'] <= dt) & (event_df['dt'] >= dt_prev_20)].sort_values(
        by='date_time', ascending=False)

    if len(rel_event) == 0:
        rel_event_list = []
        rel_event_id_list = []
    else:
        choose_dt = rel_event['dt'].iloc[0]
        rel_event = rel_event[rel_event['dt'] == choose_dt]
        rel_event_list = rel_event['event_embedding'].to_list()
        rel_event_id_list = rel_event['id'].to_list()

    rel_news = news_df[
        news_df['kdcodes'].str.contains(kdcode) & (news_df['dt'] <= dt) & (news_df['dt'] >= dt_prev_3)].sort_values(
        by='date_time', ascending=False)

    if len(rel_news) == 0:
        rel_news_list = []
        rel_news_id_list = []
    else:
        choose_dt = rel_news['dt'].iloc[0]
        rel_news = rel_news[rel_news['dt'] == choose_dt]
        rel_news_list = rel_news['news_embedding'].to_list()
        rel_news_id_list = rel_news['id'].to_list()

    relation = company_relation[company_relation['kdcode'] == kdcode]['relation_kdcode']
    if len(relation) != 0:
        relation = relation.iloc[0].split('|')
        rel_price = price_df[price_df['kdcode'].isin(relation) & (price_df['dt'] == dt)]
        rel_price = rel_price['return_embedding'].to_list()
    else:
        rel_price = []
    rel_price.append(self_price_embedding)
    
    if len(rel_event_list) == 0:
        rel_event_list == np.zeros(EVENT_NEWS_EMBEDDING_LENGTH)
    else:
        rel_event_list = np.array(rel_event_list).reshape(-1)

    if len(rel_news_list) == 0:
        rel_news_list == np.zeros(EVENT_NEWS_EMBEDDING_LENGTH)
    else:
        rel_news_list = np.array(rel_news_list).reshape(-1)
        
    return [rel_event_list, rel_event_id_list, rel_news_list, rel_news_id_list, rel_price]

def init_company_embedding_data():
    for i in range(len(ALL_QUARTER)):
        if i == 0:
            continue
        event_embedding_df, news_embedding_df = get_event_news_embedding_df()
        event_embedding_df['kdcodes'] = event_embedding_df['related_asset_code']
        news_embedding_df['kdcodes'] = news_embedding_df['rel_org_a_companies_code']

        price_embedding_df = get_price_embedding_df(ALL_QUARTER[i])
        pass
        # event_embedding_df = event_embedding_df[event_embedding_df['kdcodes'].isin(hs_300)]
        # news_embedding_df = news_embedding_df[news_embedding_df['kdcode'].isin(hs_300)]
        # price_embedding_df = price_embedding_df[price_embedding_df['kdcode'].isin(hs_300)]

        price_embedding_df[
            ['event_embedding_list', 'event_id_list', 'news_embedding_list', 'news_id_list', 'price_embedding_list']] = \
            price_embedding_df.apply(lambda x: pd.Series(get_relation_embedding_by_relation(x['kdcode'],
                                                                                            x['dt'],
                                                                                            event_embedding_df,
                                                                                            news_embedding_df,
                                                                                            price_embedding_df,
                                                                                            x['return_embedding'])),
                                     axis=1)

        price_embedding_df.to_pickle('-'.join(ALL_QUARTER[i]) + '-new:all_embedding-20-3-one-dt.pkl')
        
"""get attention event tensor
        Args:            
            embedding_events: pretrained embeddings for events
            embedding_self_price: pretrained emebdding for stock self price
            event_attention_weights: weights for source projection
            event_price_attention_weights: weights for target projection
            event_attention_alpha: graph attention for events                      

        Returns:
            attention_event: event attention result considering Inter-modality attention
            events_price_concat_embedding_e: inner-modality attention coefficient for source and target
            
        """
def get_attention_event_tensor_inner(embedding_events,
                                     embedding_self_price,
                                     event_attention_weights,
                                     event_price_attention_weights,
                                     event_attention_alpha):
    embedding_events_attention_tensor = tf.matmul(embedding_events, event_attention_weights)

    embedding_self_price_attention_tensor = tf.matmul(embedding_self_price, event_price_attention_weights)

    embedding_self_price_attention_tensor = tf.tile(embedding_self_price_attention_tensor,
                                                    [tf.shape(embedding_events_attention_tensor)[0], 1])

    events_price_concat_embedding = tf.concat(
        [embedding_events_attention_tensor, embedding_self_price_attention_tensor], 1)

    events_price_concat_embedding_e = tf.nn.softmax(
        tf.nn.leaky_relu(tf.matmul(events_price_concat_embedding, event_attention_alpha)), 0)

    attention_event = tf.reduce_sum(tf.multiply(embedding_events_attention_tensor, events_price_concat_embedding_e), 0)
    return attention_event, events_price_concat_embedding_e

"""get attention event tensor
    THIS FUNCTION DEALS WITH STOCKS THAT ARE ISOLATE
        Args:            
            embedding_events: pretrained embeddings for events
            embedding_self_price: pretrained emebdding for stock self price
            event_attention_weights: weights for source projection
            event_price_attention_weights: weights for target projection
            event_attention_alpha: graph attention for events                      

        Returns:
            attention_event: event attention result considering Inter-modality attention
            events_price_concat_embedding_e: inner-modality attention coefficient for source and target
            
        """
def get_attention_event_tensor(embedding_events,
                               embedding_self_price,
                               event_attention_weights,
                               event_price_attention_weights,
                               event_attention_alpha):
    attention_event, events_price_concat_embedding_e = tf.cond(tf.equal(tf.shape(embedding_events)[0], tf.constant(0)),
                                                               lambda: (
                                                                   tf.constant(0., shape=[1, MIDDLE_ALPHA_LENGTH]),
                                                                   tf.constant(0.0, shape=[0])),
                                                               lambda: get_attention_event_tensor_inner(
                                                                   embedding_events,
                                                                   embedding_self_price,
                                                                   event_attention_weights,
                                                                   event_price_attention_weights,
                                                                   event_attention_alpha))

    return attention_event, events_price_concat_embedding_e

"""get attention news tensor
        Args:            
            embedding_news: pretrained embeddings for news
            embedding_self_price: pretrained emebdding for stock self price
            news_attention_weights: weights for source projection
            news_price_attention_weights: weights for target projection
            news_attention_alpha: graph attention for newss                     

        Returns:
            attention_news: news attention result considering Inter-modality attention
            news_price_concat_embedding_e: inner-modality attention coefficient for source and target
            
        """        
def get_attention_news_tensor_inner(embedding_news,
                                    embedding_self_price,
                                    news_attention_weights,
                                    news_price_attention_weights,
                                    news_attention_alpha):
    embedding_news_attention_tensor = tf.matmul(embedding_news, news_attention_weights)

    embedding_self_price_attention_tensor = tf.matmul(embedding_self_price, news_price_attention_weights)

    embedding_self_price_attention_tensor = tf.tile(embedding_self_price_attention_tensor,
                                                    [tf.shape(embedding_news_attention_tensor)[0], 1])

    news_price_concat_embedding = tf.concat([embedding_news_attention_tensor, embedding_self_price_attention_tensor], 1)

    news_price_concat_embedding_e = tf.nn.softmax(
        tf.nn.leaky_relu(tf.matmul(news_price_concat_embedding, news_attention_alpha)), 0)


    attention_news = tf.reduce_sum(tf.multiply(embedding_news_attention_tensor, news_price_concat_embedding_e), 0)
    return attention_news, news_price_concat_embedding_e

"""get attention news tensor
    THIS FUNCTION DEALS WITH STOCKS THAT ARE ISOLATE
        Args:            
            embedding_news: pretrained embeddings for news
            embedding_self_price: pretrained emebdding for stock self price
            news_attention_weights: weights for source projection
            news_price_attention_weights: weights for target projection
            news_attention_alpha: graph attention for newss                     

        Returns:
            attention_news: news attention result considering Inter-modality attention
            news_price_concat_embedding_e: inner-modality attention coefficient for source and target
            
        """  
def get_attention_news_tensor(embedding_news,
                              embedding_self_price,
                              news_attention_weights,
                              news_price_attention_weights,
                              news_attention_alpha):
    attention_news, news_price_concat_embedding_e = tf.cond(tf.equal(tf.shape(embedding_news)[0], tf.constant(0)),
                                                            lambda: (tf.constant(0., shape=[1, MIDDLE_ALPHA_LENGTH]),
                                                                     tf.constant(0., shape=[0])),
                                                            lambda: get_attention_news_tensor_inner(embedding_news,
                                                                                                    embedding_self_price,
                                                                                                    news_attention_weights,
                                                                                                    news_price_attention_weights,
                                                                                                    news_attention_alpha))
    return attention_news, news_price_concat_embedding_e


"""get attention price tensor
        Args:            
            embedding_price: pretrained embeddings for price
            embedding_self_price: pretrained emebdding for stock self price
            price_attention_weights: weights for source projection
            price_price_attention_weights: weights for target projection
            price_attention_alpha: graph attention for prices                     

        Returns:
            attention_price: price attention result considering Inter-modality attention
            price_price_concat_embedding_e: inner-modality attention coefficient for source and target
            
        """  
def get_attention_price_tensor_inner(embedding_prices,
                                     embedding_self_price,
                                     price_attention_weights,
                                     price_attention_alpha):
    embedding_prices_attention_tensor = tf.matmul(embedding_prices, price_attention_weights)


    embedding_self_price_attention_tensor = tf.matmul(embedding_self_price, price_attention_weights)

    embedding_self_price_attention_tensor = tf.tile(embedding_self_price_attention_tensor,
                                                    [tf.shape(embedding_prices_attention_tensor)[0], 1])

    prices_concat_embedding = tf.concat([embedding_prices_attention_tensor, embedding_self_price_attention_tensor], 1)

    prices_concat_embedding_e = tf.nn.softmax(
        tf.nn.leaky_relu(tf.matmul(prices_concat_embedding, price_attention_alpha)), 0)


    attention_price = tf.reduce_sum(tf.multiply(embedding_prices_attention_tensor, prices_concat_embedding_e), 0)
    return attention_price

"""get attention price tensor
   EACH STOCK IS RELATED TO THE WHOLE MARKET, SO THERE'S NO NEED FOR ISOLATION DEALING PROCESS
        Args:            
            embedding_price: pretrained embeddings for price
            embedding_self_price: pretrained emebdding for stock self price
            price_attention_weights: weights for source projection
            price_price_attention_weights: weights for target projection
            price_attention_alpha: graph attention for prices                     

        Returns:
            attention_price: price attention result considering Inter-modality attention
            price_price_concat_embedding_e: inner-modality attention coefficient for source and target
            
        """  
def get_attention_price_tensor(embedding_prices,
                               embedding_self_price,
                               price_attention_weights,
                               price_attention_alpha):
    attention_price = get_attention_price_tensor_inner(embedding_prices,
                                                       embedding_self_price,
                                                       price_attention_weights,
                                                       price_attention_alpha)
    return attention_price


"""The following five functions are used to generate attention coefficient of modalities.
    We classified them in order to avoid isolated node calculations
    """  
def get_all_attention_e_1(attention_event, embedding_event, attention_news, embedding_news, attention_price,
                          final_attention_weights,
                          final_attantion_alpha):
    final_attention_e = tf.cond(tf.equal(tf.shape(embedding_news)[0], tf.constant(0)),
                                lambda: tf.constant([[0.], [0.], [1.]]),
                                lambda: get_all_attention_e_2(attention_event, embedding_event, attention_news,
                                                              embedding_news, attention_price, final_attention_weights,
                                                              final_attantion_alpha))
    return final_attention_e


def get_all_attention_e_2(attention_event, embedding_event, attention_news, embedding_news, attention_price,
                          final_attention_weights,
                          final_attantion_alpha):

    event_news_price_attention_tensor_concat = tf.concat([attention_news, attention_price], 0)
    event_news_price_attention_tensor_concat_embedding = tf.matmul(event_news_price_attention_tensor_concat,
                                                                   final_attention_weights)

    final_attention_e = tf.nn.softmax(
        tf.nn.leaky_relu(
            tf.matmul(event_news_price_attention_tensor_concat_embedding, final_attantion_alpha)), 0)

    final_attention_e = tf.concat([[[0.]], [final_attention_e[0]], [final_attention_e[1]]], 0)

    return final_attention_e


def get_all_attention_e_3(attention_event, embedding_event, attention_news, embedding_news, attention_price,
                          final_attention_weights,
                          final_attantion_alpha):
    final_attention_e = tf.cond(tf.equal(tf.shape(embedding_news)[0], tf.constant(0)),
                                lambda: get_all_attention_e_4(attention_event, embedding_event, attention_news,
                                                              embedding_news, attention_price, final_attention_weights,
                                                              final_attantion_alpha),
                                lambda: get_all_attention_e_5(attention_event, embedding_event, attention_news,
                                                              embedding_news, attention_price, final_attention_weights,
                                                              final_attantion_alpha))

    return final_attention_e


def get_all_attention_e_4(attention_event, embedding_event, attention_news, embedding_news, attention_price,
                          final_attention_weights,
                          final_attantion_alpha):

    event_news_price_attention_tensor_concat = tf.concat([attention_event, attention_price], 0)
    event_news_price_attention_tensor_concat_embedding = tf.matmul(event_news_price_attention_tensor_concat,
                                                                   final_attention_weights)

    final_attention_e = tf.nn.softmax(
        tf.nn.leaky_relu(
            tf.matmul(event_news_price_attention_tensor_concat_embedding, final_attantion_alpha)), 0)

    final_attention_e = tf.concat([[final_attention_e[0]], [[0.]], [final_attention_e[1]]], 0)

    return final_attention_e


def get_all_attention_e_5(attention_event, embedding_event, attention_news, embedding_news, attention_price,
                          final_attention_weights,
                          final_attantion_alpha):

    event_news_price_attention_tensor_concat = tf.concat([attention_event, attention_news, attention_price], 0)
    event_news_price_attention_tensor_concat_embedding = tf.matmul(event_news_price_attention_tensor_concat,
                                                                   final_attention_weights)

    final_attention_e = tf.nn.softmax(
        tf.nn.leaky_relu(
            tf.matmul(event_news_price_attention_tensor_concat_embedding, final_attantion_alpha)), 0)
    return final_attention_e

"""get all attention weights for magnn model
        Args:         
            attention_event: InnGAT results for events
            embedding_event: embeddings for events
            attention_news: InnGAT results for news
            embedding_news: embeddings for news
            attention_price: InnGAT results for price
            final_attention_weights: shared linear transformation weight matrix
            final_attantion_alpha: multi-source attention vector
                          
        Returns:
            final_attantion_tensor: representation of the target node (REPs)
            final_attention_e: attention coefficent of IntGAT
            
        """  
def get_all_attention_tensor(attention_event, 
                             embedding_event, 
                             attention_news, 
                             embedding_news, 
                             attention_price,
                             final_attention_weights,
                             final_attantion_alpha):
    attention_event = tf.reshape(attention_event, [1, MIDDLE_ALPHA_LENGTH])
    attention_news = tf.reshape(attention_news, [1, MIDDLE_ALPHA_LENGTH])
    attention_price = tf.reshape(attention_price, [1, MIDDLE_ALPHA_LENGTH])
    final_attention_e = tf.cond(tf.equal(tf.shape(embedding_event)[0], tf.constant(0)),
                                lambda: get_all_attention_e_1(attention_event, embedding_event, attention_news,
                                                              embedding_news, attention_price, final_attention_weights,
                                                              final_attantion_alpha),
                                lambda: get_all_attention_e_3(attention_event, embedding_event, attention_news,
                                                              embedding_news, attention_price, final_attention_weights,
                                                              final_attantion_alpha))

    event_news_price_attention_tensor_concat = tf.concat([attention_event, attention_news, attention_price], 0)
    event_news_price_attention_tensor_concat_embedding = tf.matmul(event_news_price_attention_tensor_concat,
                                                                   final_attention_weights)
    final_attantion_tensor = tf.reduce_sum(
        tf.multiply(event_news_price_attention_tensor_concat_embedding, final_attention_e), 0)

    return final_attantion_tensor, final_attention_e


"""doing magnn training process
        Args:
            final_attention: attention from Inter-modality source (IntSAT)
            out_weights: matrix weights
            out_biases: matrix bias

        Returns:
            results:    prediction 
            probability:   probabilities for prediction
            
        """
def get_class_vector(final_attention, out_weights, out_biases):
    final_attention = tf.reshape(final_attention, [-1, FINAL_ALPHA_LENGTH])
    results = tf.matmul(final_attention, out_weights) + out_biases
    probability  = tf.nn.softmax(results)
    return results, probability

"""doing magnn training process
        Args:
            train_period tuples in form of ('YYYY-MM-DD','YYYY-MM-DD'), which is defined in dataset/constant.py
            test_period_tuples: in form of ('YYYY-MM-DD','YYYY-MM-DD'), which is defined in dataset/constant.py
            timestamp: used for saving magnn results 

        Returns:
            
        """
def train_magnn_embedding(train_period_tuples, test_period_tuple, timestamp):    
    result_path = MAGNN_RESULT
    
    tf.reset_default_graph()
    epoch_size = 3
    
    # get train&test dataframes
    train_all_embedding_df = get_all_embedding_merge_data([train_period_tuples])
    test_all_embedding_df = get_all_embedding_merge_data([test_period_tuple])
    
    # get embedding features for traning 
    train_quarter_price_features = train_all_embedding_df[
        ['event_embedding_list', 'news_embedding_list', 'price_embedding_list', 'price_embedding']].values
    train_quarter_labels = train_all_embedding_df['label'].values
    train_quarter_labels = train_quarter_labels.astype(int)
    one_hot_train_quarter_labels = np.zeros((train_quarter_labels.size, train_quarter_labels.max() + 1))
    one_hot_train_quarter_labels[np.arange(train_quarter_labels.size), train_quarter_labels] = 1
    one_hot_train_quarter_labels = one_hot_train_quarter_labels.astype(int)
    train_dataset_array = np.concatenate([train_quarter_price_features, one_hot_train_quarter_labels], axis=1)


    batch_size = int(train_quarter_price_features.shape[0] / 4)
    batch_count = int(train_quarter_price_features.shape[0] / batch_size)

    train_dataset = MAGNN_Dataset(train_dataset_array, batch_size)

    test_quarter_price_features = test_all_embedding_df[
        ['event_embedding_list', 'news_embedding_list', 'price_embedding_list', 'price_embedding']].values
    test_quarter_labels = test_all_embedding_df['label'].values
    test_quarter_labels = test_quarter_labels.astype(int)
    one_hot_test_quarter_labels = np.zeros((test_quarter_labels.size, test_quarter_labels.max() + 1))
    one_hot_test_quarter_labels[np.arange(test_quarter_labels.size), test_quarter_labels] = 1
    one_hot_test_quarter_labels = one_hot_test_quarter_labels.astype(int)
    
    # 输入格式
    embedding_events = tf.placeholder(tf.float32)
    embedding_news = tf.placeholder(tf.float32)
    embedding_prices = tf.placeholder(tf.float32)
    embedding_self_price = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES])    
    pred_y = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # define network weights
    weights = {
        'event_attention_weights': tf.Variable(
            tf.truncated_normal([EVENT_NEWS_EMBEDDING_LENGTH, MIDDLE_ALPHA_LENGTH], 0, 0.01)),
        'event_price_attention_weights': tf.Variable(
            tf.truncated_normal([PRICE_EMBEDDING_LENGTH, MIDDLE_ALPHA_LENGTH], 0, 1)),
        'event_attention_alpha': tf.Variable(tf.truncated_normal([MIDDLE_ALPHA_LENGTH * 2, 1], 0, 1)),
        'news_attention_weights': tf.Variable(
            tf.truncated_normal([EVENT_NEWS_EMBEDDING_LENGTH, MIDDLE_ALPHA_LENGTH], 0, 0.01)),
        'news_price_attention_weights': tf.Variable(
            tf.truncated_normal([PRICE_EMBEDDING_LENGTH, MIDDLE_ALPHA_LENGTH], 0, 1)),
        'news_attention_alpha': tf.Variable(tf.truncated_normal([MIDDLE_ALPHA_LENGTH * 2, 1], 0, 1)),
        'price_attention_weights': tf.Variable(
            tf.truncated_normal([PRICE_EMBEDDING_LENGTH, MIDDLE_ALPHA_LENGTH], 0, 0.1)),
        'price_attention_alpha': tf.Variable(tf.truncated_normal([MIDDLE_ALPHA_LENGTH * 2, 1], 0, 1)),
        'final_attention_weights': tf.Variable(tf.truncated_normal([MIDDLE_ALPHA_LENGTH, FINAL_ALPHA_LENGTH], 0, 1)),
        'final_attention_alpha': tf.Variable(tf.truncated_normal([FINAL_ALPHA_LENGTH, 1], 0, 1)),
        'out_weights': tf.Variable(tf.truncated_normal([FINAL_ALPHA_LENGTH, NUM_CLASSES], 0, 1)),
        'out_bias': tf.Variable(tf.constant(0.0, shape=[NUM_CLASSES]))
    }

    attention_event, events_price_concat_embedding_e = get_attention_event_tensor(embedding_events=embedding_events,
                                                                                  embedding_self_price=embedding_self_price,
                                                                                  event_attention_weights=weights[
                                                                                      'event_attention_weights'],
                                                                                  event_price_attention_weights=weights[
                                                                                      'event_price_attention_weights'],
                                                                                  event_attention_alpha=weights[
                                                                                      'event_attention_alpha'])

    attention_news, news_price_concat_embedding_e = get_attention_news_tensor(embedding_news=embedding_news,
                                                                              embedding_self_price=embedding_self_price,
                                                                              news_attention_weights=weights[
                                                                                  'news_attention_weights'],
                                                                              news_price_attention_weights=weights[
                                                                                  'news_price_attention_weights'],
                                                                              news_attention_alpha=weights[
                                                                                  'news_attention_alpha'])

    attention_price = get_attention_price_tensor(embedding_prices=embedding_prices,
                                                 embedding_self_price=embedding_self_price,
                                                 price_attention_weights=weights['price_attention_weights'],
                                                 price_attention_alpha=weights['price_attention_alpha'])

    all_attention_tensor, final_attention_e = get_all_attention_tensor(attention_event,
                                                                       embedding_events,
                                                                       attention_news,
                                                                       embedding_news,
                                                                       attention_price,
                                                                       weights['final_attention_weights'],
                                                                       weights['final_attention_alpha'])


    # 定义损失函数和优化器，采用AdamOptimizer优化器
    pred, pred_probability = get_class_vector(all_attention_tensor, weights['out_weights'], weights['out_bias'])
    # cost = tf.reduce_mean(tf.losses.mean_squared_error(y, pred_probability))
    
    # cost = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y)) + tf.contrib.layers.apply_regularization(
    #     tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)

    # cost_v2 = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_y, labels=y)) + tf.contrib.layers.apply_regularization(
    #     tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
    cost_v2 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_y, labels=y))
    # train_op_v2 = tf.train.AdamOptimizer(lr).minimize(cost_v2)

    # 定义模型预测结果及准确率计算方法
    correct_pred = tf.equal(tf.argmax(pred_y, 1), tf.argmax(y, 1))
    accuracy_func = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    """Computes 3 different f1 scores, micro macro
        weighted.
        micro: f1 score accross the classes, as 1
        macro: mean of f1 scores per class
        weighted: weighted average of f1 scores per class,
                weighted from the support of each class


        Args:
            y_true (Tensor): labels, with shape (batch, num_classes)
            y_pred (Tensor): model's predictions, same shape as y_true

        Returns:
            tuple(Tensor): (micro, macro, weighted)
                        tuple of the computed f1 scores
        """
    def tf_f1_score(y_true, y_pred):
        f1s = [0, 0, 0]
        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)

        for i, axis in enumerate([None, 0]):
            TP = tf.count_nonzero(y_pred * y_true, axis=axis)
            FP = tf.count_nonzero(y_pred * (y_true - 1), axis=axis)
            FN = tf.count_nonzero((y_pred - 1) * y_true, axis=axis)

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)

            f1s[i] = tf.reduce_mean(f1)

        weights = tf.reduce_sum(y_true, axis=0)
        weights /= tf.reduce_sum(weights)

        f1s[2] = tf.reduce_sum(f1 * weights)

        micro, macro, weighted = f1s
        return micro, macro, weighted
    
    micro, macro, weighted = tf_f1_score(y, pred_y)

    """define batch training processes
        Args:            
            sess: tf.Session()
            features: news&event embedding lists
            labels: stock labels

        Returns:
            
        """  
    def train_batch(sess, features, labels):
        pred_results = []        
        try:
            for i in range(features.shape[0]):
                sess.run([train_op], feed_dict={
                    embedding_events: np.array([features[i][0]][0]),
                    embedding_news: np.array([features[i][1]][0]),
                    embedding_prices: np.array([features[i][2]][0]),
                    embedding_self_price: np.array([features[i][3]]),
                    y: [labels[i]]})
        except Exception as e:
            print(e)
        """get score for each epoch
        Args:            
            sess: tf.Session()
            features: news&event embedding lists
            labels: stock labels

        Returns:
            cost_value: cost value for cross-entropy
        """      
    def get_score(sess, features, labels):
        pred_probabilities = []
        for i in range(features.shape[0]):
            pred_result, pred_probability_result = sess.run([pred, pred_probability], feed_dict={
                embedding_events: np.array([features[i][0]][0]),
                embedding_news: np.array([features[i][1]][0]),
                embedding_prices: np.array([features[i][2]][0]),
                embedding_self_price: np.array([features[i][3]]),
                y: [labels[i]]})
            one_hot_result = [0, 0, 0]
            one_hot_result[np.argmax(pred_probability_result[0])] = 1
            pred_probabilities.append(one_hot_result)
            # pred_probabilities.append(pred_probability_result[0])

        cost_value, accuracy_value, micro_value, macro_value, weighted_value = sess.run(
            [cost_v2, accuracy_func, micro, macro, weighted], feed_dict={
                pred_y: pred_probabilities,
                y: labels
            })
        return cost_value, accuracy_value, micro_value, macro_value, weighted_value

    with tf.Session() as sess:
        # 初始化参数
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        step = 0
        accuracy = 0
        convergency_count = 0
        # train_loss_info = 'train loss'
        # train_accuracy_info = 'train accuracy'
        # train_micro_info = 'train micro'
        # train_macro_info = 'train macro'
        # train_weighted_info = 'train weightd'
        # test_lost_info = 'test loss'
        # test_accuracy_info = 'test accuracy'
        # test_micro_info = 'test micro'
        # test_macro_info = 'test macro'
        # test_weighted_info = 'test weightd'


        while step < epoch_size:
            train_dataset.random_shuffle()
            batch_num = 0                        
            while batch_num < batch_count:
                batch_datas, batch_labels = train_dataset.get_batch()
                train_batch(sess, batch_datas, batch_labels)
                batch_num += 1
            
            
            train_cost, train_accuracy, train_micro, train_macro, train_weighted = get_score(sess,
                                                                                             train_quarter_price_features,
                                                                                             one_hot_train_quarter_labels)

            test_cost, test_accuracy, test_micro, test_macro, test_weighted = get_score(sess,
                                                                                        test_quarter_price_features,
                                                                                        one_hot_test_quarter_labels)
            train_loss_info = 'epoch train loss: {train_cost}'.format(train_cost=train_cost)
            train_accuracy_info = 'epoch train accuracy: {train_accuracy}'.format(train_accuracy=train_accuracy)
            train_micro_info = 'epoch train micro: {train_micro}'.format(train_micro=train_micro)
            train_macro_info = 'epoch train macro: {train_macro}'.format(train_macro=train_macro)
            train_weighted_info = 'epoch train weightd: {train_weighted}'.format(train_weighted=train_weighted)
            test_loss_info = 'epoch test loss: {test_cost}'.format(test_cost=test_cost)
            test_accuracy_info = 'epoch test accuracy: {test_accuracy}'.format(test_accuracy=test_accuracy)
            test_micro_info = 'epoch test micro: {test_micro}'.format(test_micro=test_micro)
            test_macro_info = 'epoch test macro: {test_macro}'.format(test_macro=test_macro)
            test_weighted_info = 'epoch test weightd: {test_weighted}'.format(test_weighted=test_weighted)

            print('epoch: ',step)
            print(train_loss_info)
            print(train_accuracy_info)
            print(train_micro_info)
            print(train_macro_info)
            print(train_weighted_info)
            
            print(test_loss_info)
            print(test_accuracy_info)
            print(test_micro_info)
            print(test_macro_info)
            print(test_weighted_info)
            print('\n')
            
            if test_accuracy - accuracy < 0.0005:
                convergency_count += 1
                if convergency_count >= 5:
                    break
            accuracy = test_accuracy
            step += 1            


        labels = []
        probabilities = []
        events_alphas = []
        news_alphas = []
        final_alphas = []

        # testing for magnn
        for i in range(test_quarter_price_features.shape[0]):
            probability, \
            events_alpha, \
            news_alpha, \
            final_alpha = sess.run(
                [pred_probability,
                 events_price_concat_embedding_e,
                 news_price_concat_embedding_e,
                 final_attention_e],
                feed_dict={embedding_events: np.array([test_quarter_price_features[i][0]][0]),
                           embedding_news: np.array([test_quarter_price_features[i][1]][0]),
                           embedding_prices: np.array([test_quarter_price_features[i][2]][0]),
                           embedding_self_price: np.array([test_quarter_price_features[i][3]]),
                           y: [one_hot_test_quarter_labels[i]]})                     
            labels.append(np.argmax(probability))
            probabilities.append(probability[0])
            events_alphas.append(events_alpha.reshape(-1))
            news_alphas.append(news_alpha.reshape(-1))
            final_alphas.append(final_alpha.reshape(-1))
                                
        if not os.path.exists(os.path.join(result_path, timestamp)):
            os.mkdir(os.path.join(result_path, timestamp))
            
        # save results for magnn
        np.save(os.path.join(result_path, timestamp, '-'.join(
            test_period_tuple) + '-magnn_label-test-{lr}-{PRICE_EMBEDDING_LENGTH}-{MIDDLE_ALPHA_LENGTH}-{FINAL_WEIGHT_LENGTH}-{FINAL_ALPHA_LENGTH}'.format(
            lr=lr,
            PRICE_EMBEDDING_LENGTH=PRICE_EMBEDDING_LENGTH,
            MIDDLE_ALPHA_LENGTH=MIDDLE_ALPHA_LENGTH,
            FINAL_WEIGHT_LENGTH=FINAL_WEIGHT_LENGTH,
            FINAL_ALPHA_LENGTH=FINAL_ALPHA_LENGTH)), np.array(labels).reshape([-1, 1]))
        np.save(os.path.join(result_path, timestamp, '-'.join(
            test_period_tuple) + '-magnn_probability-test-{lr}-{PRICE_EMBEDDING_LENGTH}-{MIDDLE_ALPHA_LENGTH}-{FINAL_WEIGHT_LENGTH}-{FINAL_ALPHA_LENGTH}'.format(
            lr=lr,
            PRICE_EMBEDDING_LENGTH=PRICE_EMBEDDING_LENGTH,
            MIDDLE_ALPHA_LENGTH=MIDDLE_ALPHA_LENGTH,
            FINAL_WEIGHT_LENGTH=FINAL_WEIGHT_LENGTH,
            FINAL_ALPHA_LENGTH=FINAL_ALPHA_LENGTH
        )), np.array(probabilities))
        
        np.save(os.path.join(result_path, timestamp, '-'.join(
            test_period_tuple) + '-magnn_event_alpha-test-{lr}-{PRICE_EMBEDDING_LENGTH}-{MIDDLE_ALPHA_LENGTH}-{FINAL_WEIGHT_LENGTH}-{FINAL_ALPHA_LENGTH}'.format(
            lr=lr,
            PRICE_EMBEDDING_LENGTH=PRICE_EMBEDDING_LENGTH,
            MIDDLE_ALPHA_LENGTH=MIDDLE_ALPHA_LENGTH,
            FINAL_WEIGHT_LENGTH=FINAL_WEIGHT_LENGTH,
            FINAL_ALPHA_LENGTH=FINAL_ALPHA_LENGTH)), np.array(events_alphas))
        np.save(os.path.join(result_path, timestamp, '-'.join(
            test_period_tuple) + '-magnn_news_alpha-test-{lr}-{PRICE_EMBEDDING_LENGTH}-{MIDDLE_ALPHA_LENGTH}-{FINAL_WEIGHT_LENGTH}-{FINAL_ALPHA_LENGTH}'.format(
            lr=lr,
            PRICE_EMBEDDING_LENGTH=PRICE_EMBEDDING_LENGTH,
            MIDDLE_ALPHA_LENGTH=MIDDLE_ALPHA_LENGTH,
            FINAL_WEIGHT_LENGTH=FINAL_WEIGHT_LENGTH,
            FINAL_ALPHA_LENGTH=FINAL_ALPHA_LENGTH)), np.array(news_alphas))
        np.save(os.path.join(result_path, timestamp, '-'.join(
            test_period_tuple) + '-magnn_final_alpha-test-{lr}-{PRICE_EMBEDDING_LENGTH}-{MIDDLE_ALPHA_LENGTH}-{FINAL_WEIGHT_LENGTH}-{FINAL_ALPHA_LENGTH}'.format(
            lr=lr,
            PRICE_EMBEDDING_LENGTH=PRICE_EMBEDDING_LENGTH,
            MIDDLE_ALPHA_LENGTH=MIDDLE_ALPHA_LENGTH,
            FINAL_WEIGHT_LENGTH=FINAL_WEIGHT_LENGTH,
            FINAL_ALPHA_LENGTH=FINAL_ALPHA_LENGTH)), np.array(final_alphas))

        pass
    
if __name__ == '__main__':
    import time

    timestamp = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    for i in range(1):
        test_period_tuple = TEST_QUARTERS[0]
        train_period_tuples = TEST_QUARTERS[0]
        train_magnn_embedding(train_period_tuples, test_period_tuple, timestamp)
