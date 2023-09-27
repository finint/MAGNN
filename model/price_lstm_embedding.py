import math
import os
# from qsdata.api import *
import numpy as np
import tensorflow as tf

import sys
sys.path.append('.')

from prepossessing.constant import TEST_QUARTERS
from prepossessing.magnn_dataset import MAGNN_Dataset
from prepossessing.data_prepare import get_price_segment
from tools.tools import *

from datetime import datetime

STEP_SIZE = 3 # change for usage

MEDIUM_RANGE = (-0.008, 0.008)

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

FEATURE_NUM = 4
LABEL_NUM = 3

PRICE_NUMPY_DIR = './data/price-numpy'
PRICE_EMBEDDING_DIR = './data/price-embedding'

PRICE_EMBEDDING_LOG_PATH = '../data/price-embedding-log'


# MAD:中位数去极值
def ds_filter_extreme_MAD(series, n=5, min_range=None, max_range=None):
    if min_range is not None and max_range is not None:
        return np.clip(series, min_range, max_range), min_range, max_range
    else:
        median = np.percentile(series, 50)
        new_median = np.percentile((series - median).abs(), 50)
        max_range = median + n * new_median
        min_range = median - n * new_median
        return np.clip(series, min_range, max_range), min_range, max_range


def ds_standardize_zscore(series, mean=None, std=None):
    if mean is not None and std is not None:
        return (series - mean) / std, mean, std
    else:
        std = series.std()
        mean = series.mean()
        return (series - mean) / std, mean, std


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


def get_one_day_return(close_price_ser, periods=1):
    shift_close_price_ser = close_price_ser.shift(periods=periods)
    if periods == 1:
        one_day_return = (close_price_ser - shift_close_price_ser) / shift_close_price_ser
    else:
        one_day_return = (shift_close_price_ser - close_price_ser) / close_price_ser
    return one_day_return


def get_features(x):
    features = x[[ONE_DAY_HIGH_RETURN, ONE_DAY_OPEN_RETURN, ONE_DAY_LOW_RETURN, ONE_DAY_CLOSE_RETURN]].values.reshape(
        -1, )
    return list(features)


def get_features_and_labels(data_df, start_dt, end_dt):
    trading_dates = get_trading_dates(start_dt, end_dt)
    
    features = []
    labels = []
    labels_normal = []
    dts = []
    kdcodes = []
    for dt in trading_dates:
        t1 = datetime.now()
        start_dt = str(get_previous_trading_date(dt, STEP_SIZE - 1))
        end_dt = str(dt)
        data_df_peried = data_df[(data_df['dt'] <= end_dt) & (data_df['dt'] >= start_dt)]
        features_dt = data_df_peried[data_df_peried['dt'] <= end_dt].groupby('kdcode').apply(
            lambda x: get_features(x))
        correct_index = features_dt.apply(lambda x: len(x) == FEATURE_NUM * STEP_SIZE)
        correct_index = correct_index[correct_index == True].index
        features_dt = features_dt[correct_index]
        labels_dt = data_df_peried[data_df_peried['dt'] == end_dt].set_index('kdcode')[LABEL][correct_index]
        labels_normal_dt = data_df_peried[data_df_peried['dt'] == end_dt].set_index('kdcode')[LABEL_NORMAL][correct_index]
        features.extend(features_dt.to_list())
        labels.extend(labels_dt.to_list())
        # labels_normal.extend(labels_normal_dt.to_list())
        dts.extend([dt for i in range(len(features_dt))])
        kdcodes.extend(list(features_dt.index))
        t2 = datetime.now()
    return features, labels, dts, kdcodes
    # return features, labels_normal, dts, kdcodes


def get_train_test_data(train_period_tuple, test_period_tuple):
    if os.path.exists(os.path.join(PRICE_NUMPY_DIR, '-'.join(train_period_tuple) + '-price_features-train.npy'), allow_pickle=True) \
            and os.path.exists(os.path.join(PRICE_NUMPY_DIR, '-'.join(test_period_tuple) + '-price_features-test.npy')):
        train_period_features = np.load(
            os.path.join(PRICE_NUMPY_DIR, '-'.join(train_period_tuple) + '-price_features-train.npy'), allow_pickle=True)
        train_period_labels = np.load(os.path.join(PRICE_NUMPY_DIR, '-'.join(train_period_tuple) + '-labels-train.npy'), allow_pickle=True)
        train_period_dts = np.load(os.path.join(PRICE_NUMPY_DIR, '-'.join(train_period_tuple) + '-dts-train.npy'), allow_pickle=True)
        train_period_kdcodes = np.load(
            os.path.join(PRICE_NUMPY_DIR, '-'.join(train_period_tuple) + '-kdcodes-train.npy'), allow_pickle=True)

        test_period_features = np.load(
            os.path.join(PRICE_NUMPY_DIR, '-'.join(test_period_tuple) + '-price_features-test.npy'), allow_pickle=True)
        test_period_labels = np.load(os.path.join(PRICE_NUMPY_DIR, '-'.join(test_period_tuple) + '-labels-test.npy'), allow_pickle=True)
        test_period_dts = np.load(os.path.join(PRICE_NUMPY_DIR, '-'.join(test_period_tuple) + '-dts-test.npy'), allow_pickle=True)
        test_period_kdcodes = np.load(os.path.join(PRICE_NUMPY_DIR, '-'.join(test_period_tuple) + '-kdcodes-test.npy'), allow_pickle=True)
        return train_period_features, \
               train_period_labels, \
               train_period_dts, \
               train_period_kdcodes, \
               test_period_features, \
               test_period_labels, \
               test_period_dts, \
               test_period_kdcodes

    # this is set for demonstration
    train_period_data_start_dt = train_period_tuple[0]
    train_period_data_end_dt = train_period_tuple[1]
    test_period_data_start_dt = test_period_tuple[0]
    test_period_data_end_dt = test_period_tuple[1]
    
    # ori codes:
    #train_period_data_start_dt = str(get_previous_trading_date(train_period_tuple[0], STEP_SIZE - 1))
    #test_period_data_start_dt = str(get_previous_trading_date(test_period_tuple[0], STEP_SIZE - 1))

    train_period_df = get_price_segment(train_period_data_start_dt, train_period_data_end_dt).dropna()
    test_period_df = get_price_segment(test_period_data_start_dt, test_period_data_end_dt).dropna()

    for feature in [ONE_DAY_HIGH_RETURN, ONE_DAY_OPEN_RETURN, ONE_DAY_LOW_RETURN, ONE_DAY_CLOSE_RETURN]:
        train_period_df[feature], min_range, max_range = ds_filter_extreme_MAD(train_period_df[feature])
        test_period_df[feature], _, _ = ds_filter_extreme_MAD(test_period_df[feature], 5, min_range, max_range)
        train_period_df[feature], mean, std = ds_standardize_zscore(train_period_df[feature])
        test_period_df[feature], _, _ = ds_standardize_zscore(test_period_df[feature], mean, std)

    
    train_period_features, train_period_labels, train_period_dts, train_period_kdcodes = get_features_and_labels(
        train_period_df, train_period_tuple[0], train_period_data_end_dt)
        
    
    np.save(os.path.join(PRICE_NUMPY_DIR, '-'.join(train_period_tuple) + '-price_features-train'),
            train_period_features)
    np.save(os.path.join(PRICE_NUMPY_DIR, '-'.join(train_period_tuple) + '-labels-train'), train_period_labels)
    np.save(os.path.join(PRICE_NUMPY_DIR, '-'.join(train_period_tuple) + '-dts-train'), train_period_dts)
    np.save(os.path.join(PRICE_NUMPY_DIR, '-'.join(train_period_tuple) + '-kdcodes-train'), train_period_kdcodes)

    test_period_features, test_period_labels, test_period_dts, test_period_kdcodes = get_features_and_labels(
        test_period_df, test_period_tuple[0], test_period_data_end_dt)
    np.save(os.path.join(PRICE_NUMPY_DIR, '-'.join(test_period_tuple) + '-price_features-test'), test_period_features)
    np.save(os.path.join(PRICE_NUMPY_DIR, '-'.join(test_period_tuple) + '-labels-test'), test_period_labels)
    np.save(os.path.join(PRICE_NUMPY_DIR, '-'.join(test_period_tuple) + '-dts-test'), test_period_dts)
    np.save(os.path.join(PRICE_NUMPY_DIR, '-'.join(test_period_tuple) + '-kdcodes-test'), test_period_kdcodes)

    return np.array(train_period_features), \
           np.array(train_period_labels), \
           np.array(train_period_dts), \
           np.array(train_period_kdcodes), \
           np.array(test_period_features), \
           np.array(test_period_labels), \
           np.array(test_period_dts), \
           np.array(test_period_kdcodes)


def train_price_embedding(train_period_tuple, test_period_tuple):
    tf.reset_default_graph()
    # hyperparameters 超参数
    lr = 0.001
    epoch_size = 5

    # 导入数据
    train_quarter_price_features, train_quarter_labels, _, _, \
    test_quarter_price_features, test_quarter_labels, _, _ = get_train_test_data(
        train_period_tuple, test_period_tuple)
    
    train_quarter_labels = train_quarter_labels.astype(int)
    one_hot_train_quarter_labels = np.zeros((train_quarter_labels.size, train_quarter_labels.max() + 1))
    one_hot_train_quarter_labels[np.arange(train_quarter_labels.size), train_quarter_labels] = 1
    one_hot_train_quarter_labels = one_hot_train_quarter_labels.astype(int)
    train_dataset_array = np.concatenate([train_quarter_price_features, one_hot_train_quarter_labels], axis=1)
    
    np.save('array',train_dataset_array)
    
    batch_size = int(train_quarter_price_features.shape[0])
    batch_count = int(train_quarter_price_features.shape[0] / batch_size)

    train_dataset = MAGNN_Dataset(train_dataset_array, batch_size)

    test_quarter_labels = test_quarter_labels.astype(int)
    one_hot_test_quarter_labels = np.zeros((test_quarter_labels.size, test_quarter_labels.max() + 1))
    one_hot_test_quarter_labels[np.arange(test_quarter_labels.size), test_quarter_labels] = 1
    one_hot_test_quarter_labels = one_hot_test_quarter_labels.astype(int)

    n_inputs = FEATURE_NUM
    n_steps = STEP_SIZE
    n_hidden_units = 64  # 隐藏层神经元数目
    # 用于分类模型
    n_classes = 3  # classes(0-3 digits)

    # 输入
    x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # 定义权值
    weights = {
        # (n_hidden_units, 3)
        'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
    }
    biases = {
        # (3, )
        'out': tf.Variable(tf.constant(0.01, shape=[n_classes]))
    }

    def RNN(X, weights, biases):
        # X (batch size * steps, inputs)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        # lstm cell 被分成两部分，(c_state, m_state)
        _init_state = lstm_cell.zero_state(tf.shape(X)[0], dtype=tf.float32)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, X, initial_state=_init_state, time_major=False)
        # 用于分类模型
        pred_results = tf.matmul(states[1], weights['out']) + biases['out']
        pred_probability = tf.nn.softmax(pred_results)
        return pred_results, states[1], pred_probability

    # 定义损失函数和优化器，采用AdamOptimizer优化器
    pred, embedding, probability = RNN(x, weights, biases)
    # cost = tf.reduce_mean(tf.losses.mean_squared_error(y, probability))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    
    # 定义模型预测结果及准确率计算方法
    correct_pred = tf.equal(tf.argmax(probability, 1), tf.argmax(y, 1))
    accuracy_func = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # init = tf.initialize_all_variables()

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

    # def get_score(sess, features, labels):
    #     cost_value = sess.run(
    #         cost, feed_dict={
    #             x: features,
    #             y: labels,
    #         })
    #     return cost_value
    
    def get_score(sess, features, labels):
        cost_value, accuracy_value, micro_value, macro_value, weighted_value = sess.run(
            [cost, accuracy_func, micro, macro, weighted], feed_dict={
                x: features,
                y: labels,
            })
        return cost_value, accuracy_value, micro_value, macro_value, weighted_value

    with tf.Session() as sess:
        # 初始化参数
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        step = 0
        accuracy = 0
        convergency_count = 0

        train_loss_info = 'epoch train loss'
        train_accuracy_info = 'epoch train accuracy'
        train_micro_info = 'epoch train micro'
        train_macro_info = 'epoch train macro'
        train_weighted_info = 'epoch train weightd'
        test_lost_info = 'epoch test loss'
        test_accuracy_info = 'epoch test accuracy'
        test_micro_info = 'epoch test micro'
        test_macro_info = 'epoch test macro'
        test_weighted_info = 'epoch test weightd'

        while step < 1:
            train_dataset.random_shuffle()
            batch_num = 0
            while batch_num < batch_count:
                batch_datas, batch_labels = train_dataset.get_batch()
                batch_datas = batch_datas.reshape([-1, n_steps, n_inputs])
                sess.run([train_op], feed_dict={
                    x: batch_datas,
                    y: batch_labels,
                })
                batch_num += 1

            # train_loss = get_score(sess,
            #                        train_quarter_price_features.reshape(
            #                            [-1, n_steps,
            #                             n_inputs]),
            #                        train_quarter_labels)

            # test_loss = get_score(sess,
            #                       test_quarter_price_features.reshape(
            #                           [-1, n_steps, n_inputs]),
            #                       test_quarter_labels)
            
            train_cost, train_accuracy, train_micro, train_macro, train_weighted = get_score(sess,
                                                                                             train_quarter_price_features.reshape(
                                                                                                 [-1, n_steps,
                                                                                                  n_inputs]),
                                                                                             one_hot_train_quarter_labels)
            
            test_cost, test_accuracy, test_micro, test_macro, test_weighted = get_score(sess,
                                                                                        test_quarter_price_features.reshape(
                                                                                            [-1, n_steps, n_inputs]),
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
                        
            if test_accuracy - accuracy < 0.001:
                convergency_count += 1
                if convergency_count >= 5:
                    break
            accuracy = test_accuracy
            step += 1
            
        test_quarter_price_embedding, test_quarter_price_label, test_quarter_price_probability = sess.run(
            [embedding, tf.argmax(pred, 1), probability], feed_dict={
                x: test_quarter_price_features.reshape([-1, n_steps, n_inputs]),
            })
        if not os.path.exists(os.path.join(PRICE_EMBEDDING_DIR)):
            os.mkdir(os.path.join(PRICE_EMBEDDING_DIR))
        np.save(
            os.path.join(PRICE_EMBEDDING_DIR, '-'.join(test_period_tuple) + '-lstm_model_embedding-test'),
            test_quarter_price_embedding)
        np.save(os.path.join(PRICE_EMBEDDING_DIR, '-'.join(test_period_tuple) + '-lstm_model_label-test'),
                test_quarter_price_label.reshape([-1, 1]))
        np.save(
            os.path.join(PRICE_EMBEDDING_DIR, '-'.join(test_period_tuple) + '-lstm_model_probability-test'),
            test_quarter_price_probability)


def get_price_embedding_df(period_tuple):
    dts = np.load(os.path.join(PRICE_NUMPY_DIR, '-'.join(period_tuple) + '-dts-test.npy'), allow_pickle=True)
    kdcodes = np.load(os.path.join(PRICE_NUMPY_DIR, '-'.join(period_tuple) + '-kdcodes-test.npy'), allow_pickle=True)
    labels = np.load(os.path.join(PRICE_NUMPY_DIR, '-'.join(period_tuple) + '-labels-test.npy'), allow_pickle=True)
    df = pd.DataFrame({'dt': dts, 'kdcode': kdcodes, 'label': labels})
    return_embedding = np.load(os.path.join(PRICE_NUMPY_DIR, '-'.join(period_tuple) + '-lstm_model_embedding-test.npy', allow_pickle=True))
    df['price_embedding'] = return_embedding.tolist()
    return df


if __name__ == "__main__":
    train_price_embedding(TEST_QUARTERS[0], TEST_QUARTERS[0])
