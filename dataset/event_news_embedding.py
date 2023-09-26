import os
import sys
sys.path.append('.')
from dataset.constant import TEST_QUARTERS
from dataset.magnn_dataset import MAGNN_Dataset
from dataset.data_prepare import get_event_news_embedding_merge_data

from tools.tools import *

import tensorflow as tf
import numpy as np
from bert_serving.client import BertClient

FEATURE_NUM = 768
LABEL_NUM = 3
STEP_SIZE = 6
EVENT_NEWS_EMBEDDING_LENGTH = 768

START_DT = '2019-04-01'
news_df_file_path = 'news_df_{start_dt}.pkl'.format(start_dt=START_DT)
news_embedding_df_file_path = 'news_embedding_df_{start_dt}.pkl'.format(start_dt=START_DT)

event_df_file_path = 'event_df_{start_dt}.pkl'.format(start_dt=START_DT)
event_embedding_df_file_path = 'event_embedding_df_{start_dt}.pkl'.format(start_dt=START_DT)

# relative path should be modified
EVENT_NEWS_RESULT = '/data/event-news-result'
EVENT_NEWS_EMBEDDING_LOG_PATH = '/data/event-news-embedding-log'


def get_event_news_embedding_df():
    event_df, news_df=pd.DataFrame(),pd.DataFrame()
    
    if not os.path.exists(event_embedding_df_file_path) and not os.path.exists(news_embedding_df_file_path):
        if not os.path.exists(news_df_file_path):
            print(news_df_file_path,'Please get your own data from your database')
            return event_df, news_df
        else:
            news_df = pd.read_pickle(news_df_file_path)

        if not os.path.exists(event_df_file_path):  
            print(news_df_file_path,'Please get your own data from your database')
            return event_df, news_df
        else:
            event_df = pd.read_pickle(event_df_file_path)

        bc = BertClient()

        event_df = event_df.dropna()
        news_df = news_df.dropna()

        event_df['event_embedding'] = bc.encode(list(event_df['event_content'])).tolist()

        news_df['news_embedding'] = bc.encode(list(news_df['content'])).tolist()

        event_df.to_pickle(event_embedding_df_file_path)
        news_df.to_pickle(news_embedding_df_file_path)
    else:
        event_df = pd.read_pickle(event_embedding_df_file_path)
        news_df = pd.read_pickle(news_embedding_df_file_path)
    event_df['date_time'] = pd.to_datetime(event_df['event_publish_dt'])
    event_df['dt'] = event_df['date_time'].dt.date
    news_df['date_time'] = pd.to_datetime(news_df['date_time'])
    news_df['dt'] = news_df['date_time'].dt.date
    return event_df, news_df


def merge_embeddings(x):
    if len(x) == 0:
        return list(np.array([0.0 for i in range(FEATURE_NUM)]))
    else:
        return list(np.mean(x, axis=0))


def train_event_or_news_embedding(train_period_tuples, test_period_tuple, timestamp, events_or_news='events'):
    tf.reset_default_graph()
    epoch_size = 6
    lr = 0.001
    train_all_embedding_df = get_event_news_embedding_merge_data(train_period_tuples)
    train_all_embedding_df['label'] = train_all_embedding_df['label_normal']
    test_all_embedding_df = get_event_news_embedding_merge_data([test_period_tuple])
    test_all_embedding_df['label'] = test_all_embedding_df['label_normal']

    if events_or_news == 'events':
        train_all_embedding_df['feature'] = train_all_embedding_df['event_embedding_list']
        test_all_embedding_df['feature'] = test_all_embedding_df['event_embedding_list']
    elif events_or_news == 'news':
        train_all_embedding_df['feature'] = train_all_embedding_df['news_embedding_list']
        test_all_embedding_df['feature'] = test_all_embedding_df['news_embedding_list']

    train_all_embedding_df['feature'] = train_all_embedding_df['feature'].apply(lambda x: merge_embeddings(x))
    test_all_embedding_df['feature'] = test_all_embedding_df['feature'].apply(lambda x: merge_embeddings(x))

    train_all_embedding_df = train_all_embedding_df.dropna()
    test_all_embedding_df = test_all_embedding_df.dropna()

    train_quarter_price_features = np.array(train_all_embedding_df['feature'].to_list())
    test_quarter_price_features = np.array(test_all_embedding_df['feature'].to_list())

    train_quarter_labels = train_all_embedding_df['label'].values
    train_quarter_labels = train_quarter_labels.reshape([-1, 1])
    # 用于分类模型
    # train_quarter_labels = train_quarter_labels.astype(int)
    # one_hot_train_quarter_labels = np.zeros((train_quarter_labels.size, train_quarter_labels.max() + 1))
    # one_hot_train_quarter_labels[np.arange(train_quarter_labels.size), train_quarter_labels] = 1
    # one_hot_train_quarter_labels = one_hot_train_quarter_labels.astype(int)
    # train_dataset_array = np.concatenate([train_quarter_price_features, one_hot_train_quarter_labels], axis=1)
    train_dataset_array = np.concatenate([train_quarter_price_features, train_quarter_labels], axis=1)

    batch_size = int(train_quarter_price_features.shape[0] / 4)
    batch_count = int(train_quarter_price_features.shape[0] / batch_size)
    print(batch_count)

    train_dataset = MAGNN_Dataset(train_dataset_array, batch_size)
    test_quarter_labels = test_all_embedding_df['label'].values
    test_quarter_labels = test_quarter_labels.reshape([-1, 1])
    
    # 用于分类模型
    # test_quarter_labels = test_quarter_labels.astype(int)
    # one_hot_test_quarter_labels = np.zeros((test_quarter_labels.size, test_quarter_labels.max() + 1))
    # one_hot_test_quarter_labels[np.arange(test_quarter_labels.size), test_quarter_labels] = 1
    # one_hot_test_quarter_labels = one_hot_test_quarter_labels.astype(int)

    n_inputs = FEATURE_NUM
    n_steps = STEP_SIZE
    n_classes = 3  # classes(0-3 digits)
    n_classes = 1  # classes(0-3 digits)

    # 输入
    x = tf.placeholder(tf.float32, [None, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # 定义权值
    weights = {
        # (n_hidden_units, 3)
        'out': tf.Variable(tf.random_normal([n_inputs, n_classes]))
    }

    biases = {
        # (3, )
        'out': tf.Variable(tf.constant(0.001, shape=[n_classes]))
    }

    def RNN(X, weights, biases):
        pred_results = tf.matmul(X, weights['out']) + biases['out']
        # 用于分类模型
        # pred_probability = tf.nn.softmax(pred_results)
        pred_probability = tf.nn.sigmoid(pred_results)
        return pred_results, pred_probability

    # 定义损失函数和优化器，采用AdamOptimizer优化器
    # define loss function & optimizer:AdamOptimizer
    pred, probability = RNN(x, weights, biases)
    cost = tf.reduce_mean(tf.losses.mean_squared_error(y, probability))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    # 定义模型预测结果及准确率计算方法
    # define model predict result& accuracy function
    correct_pred = tf.equal(tf.argmax(probability, 1), tf.argmax(y, 1))
    accuracy_func = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def tf_f1_score(y_true, y_pred):
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

    micro, macro, weighted = tf_f1_score(y, probability)

    # 用于分类模型
    
    # define various models
    # def get_score(sess, features, labels):
    #     cost_value, accuracy_value, micro_value, macro_value, weighted_value = sess.run(
    #         [cost, accuracy_func, micro, macro, weighted], feed_dict={
    #             x: features,
    #             y: labels,
    #         })
    #     return cost_value, accuracy_value, micro_value, macro_value, weighted_value

    def get_score(sess, features, labels):
        cost_value = sess.run(
            cost, feed_dict={
                x: features,
                y: labels,
            })
        return cost_value

    with tf.Session() as sess:
        # 初始化参数
        # initialize params
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        step = 0
        accuracy = 0
        convergency_count = 0

        train_loss_info = 'train loss'
        train_accuracy_info = 'train accuracy'
        train_micro_info = 'train micro'
        train_macro_info = 'train macro'
        train_weighted_info = 'train weightd'
        test_lost_info = 'test loss'
        test_accuracy_info = 'test accuracy'
        test_micro_info = 'test micro'
        test_macro_info = 'test macro'
        test_weighted_info = 'test weightd'

        title = '\t'.join(
            [train_loss_info, train_accuracy_info, train_micro_info, train_macro_info, train_weighted_info,
             test_lost_info, test_accuracy_info, test_micro_info, test_macro_info, test_weighted_info])
        print(title)

        while step < epoch_size:
            train_dataset.random_shuffle()
            batch_num = 0
            while batch_num < batch_count:
                batch_datas, batch_labels = train_dataset.get_batch()
                sess.run([train_op], feed_dict={
                    x: batch_datas,
                    y: batch_labels,
                })
                batch_num += 1

            train_loss = get_score(sess,
                                   train_quarter_price_features,
                                   train_quarter_labels)

            test_loss = get_score(sess,
                                  test_quarter_price_features,
                                  test_quarter_labels)
            step += 1

        test_quarter_price_label, test_quarter_price_probability = sess.run(
            [tf.argmax(pred, 1), probability], feed_dict={
                x: test_quarter_price_features,
            })

        if not os.path.exists(os.path.join(EVENT_NEWS_RESULT, timestamp)):
            os.mkdir(os.path.join(EVENT_NEWS_RESULT, timestamp))
        np.save(os.path.join(EVENT_NEWS_RESULT, timestamp,
                             ':'.join(test_period_tuple) + ':{type}_embedingg_model_label'.format(type=events_or_news)),
                test_quarter_price_label.reshape([-1, 1]))
        np.save(os.path.join(EVENT_NEWS_RESULT, timestamp,
                             ':'.join(test_period_tuple) + ':{type}_embedingg_model_probability'.format(
                                 type=events_or_news)), test_quarter_price_probability)


if __name__ == "__main__":
    import time

    timestamp = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))

    for i in range(1, 6):
        test_period_tuple = TEST_QUARTERS[i]
        train_period_tuples = TEST_QUARTERS[0:i]
        # SWITCH for your own news/events trainning
        # train_event_or_news_embedding(train_period_tuples, test_period_tuple, timestamp, 'news')
        train_event_or_news_embedding(train_period_tuples, test_period_tuple, timestamp, 'events')
