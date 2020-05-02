# -*- coding:utf-8 -*-
from utils.data_loader import load_train_dataset, load_test_dataset
import tensorflow as tf


def train_batch_generator(batch_size, max_enc_len=200, max_dec_len=50, sample_sum=None):
    # 加载数据集
    train_X, train_Y = load_train_dataset(max_enc_len, max_dec_len)
    if sample_sum:
        train_X = train_Y[:sample_sum]
        train_Y = train_Y[:sample_sum]
    dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).shuffle(len(train_X))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    steps_per_epoch = len(train_X) // batch_size
    return dataset, steps_per_epoch


def bream_test_batch_generator(beam_size, max_enc_len=200):
    # 加载数据集
    test_X = load_test_dataset(max_enc_len)
    for row in test_X:
        beam_search_data = tf.convert_to_tensor([row for i in range(beam_size)])
        yield beam_search_data


if __name__ == '__main__':
    bream_test_batch_generator(4)
