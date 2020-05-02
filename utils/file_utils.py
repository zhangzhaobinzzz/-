# -*- coding:utf-8 -*-
import os

from utils.config import save_result_dir
import time
import pickle


def save_vocab(file_path, data):
    with open(file_path) as f:
        for i in data:
            f.write(i)


def get_result_filename(batch_size, epochs, max_length_inp, embedding_dim, commit=''):
    """
    获取时间
    :return:
    """
    now_time = time.strftime('%Y_%m_%d_%H_%M_%S')
    filename = now_time + '_batch_size_{}_epochs_{}_max_length_inp_{}_embedding_dim_{}{}.csv'.format(batch_size, epochs,
                                                                                                     max_length_inp,
                                                                                                     embedding_dim,
                                                                                                     commit)
    result_save_path = os.path.join(save_result_dir, filename)
    return result_save_path


def save_pickle(batch_data, pickle_path):
    f = open(pickle_path, 'wb')
    pickle.dump(batch_data, f)


def load_pickle(pickle_path):
    f = open(pickle_path, 'rb')
    batch_data = pickle.load(f)
    return batch_data


def save_dict(save_path, dict_data):
    """
    保存字典
    :param save_path: 保存路径
    :param dict_data: 字典路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        for k, v in dict_data.items():
            f.write("{}\t{}\n".format(k, v))


def load_dict(file_path):
    """
    读取字典
    :param file_path: 文件路径
    :return: 返回读取后的字典
    """
    return dict((line.strip().split("\t")[0], idx)
                for idx, line in enumerate(open(file_path, "r", encoding='utf-8').readlines()))
