# -*- coding:utf-8 -*-
import re
import jieba
import pandas as pd
from utils.file_utils import save_dict
from utils.multi_proc_utils import parallelize
from utils.config import train_seg_path, test_seg_path, merger_seg_path, user_dict, train_x_seg_path, test_x_seg_path, \
    train_x_pad_path, train_y_pad_path, test_x_pad_path, wv_train_epochs, embedding_matrix_path, \
    vocab_path, reverse_vocab_path, train_x_path, train_y_path, test_x_path, embedding_dim, train_y_seg_path, \
    val_x_seg_path, val_y_seg_path
from utils.config import stop_word_path, train_data_path, test_data_path
from gensim.models.word2vec import LineSentence, Word2Vec
import numpy as np
# 引入日志配置
import logging
from utils.config import save_wv_model_path
from sklearn.model_selection import train_test_split

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 自定义词表
jieba.load_userdict(user_dict)


def build_dataset(train_data_path, test_data_path):
    '''
    数据加载+预处理
    :param train_data_path:训练集路径
    :param test_data_path: 测试集路径
    :return: 训练数据 测试数据  合并后的数据
    '''
    # 1.加载数据
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    print('train data size {},test data size {}'.format(len(train_df), len(test_df)))

    # 2. 空值剔除
    train_df.dropna(subset=['Report'], inplace=True)

    train_df.fillna('', inplace=True)
    test_df.fillna('', inplace=True)

    # 3.多线程, 批量数据处理
    train_df = parallelize(train_df, sentences_proc)
    test_df = parallelize(test_df, sentences_proc)

    # 4. 合并训练测试集合
    train_df['merged'] = train_df[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
    test_df['merged'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    merged_df = pd.concat([train_df[['merged']], test_df[['merged']]], axis=0)
    print('train data size {},test data size {},merged_df data size {}'.format(len(train_df),
                                                                               len(test_df),
                                                                               len(merged_df)))

    # 5.保存处理好的 训练 测试集合
    train_df = train_df.drop(['merged'], axis=1)
    test_df = test_df.drop(['merged'], axis=1)

    train_df.to_csv(train_seg_path, index=None, header=False)
    test_df.to_csv(test_seg_path, index=None, header=False)

    # 6. 保存合并数据
    merged_df.to_csv(merger_seg_path, index=None, header=False)

    # 7. 训练词向量
    print('start build w2v model')
    wv_model = Word2Vec(LineSentence(merger_seg_path),
                        size=embedding_dim,
                        sg=1,
                        workers=8,
                        iter=wv_train_epochs,
                        window=5,
                        min_count=5)

    # 8. 分离数据和标签
    train_df['X'] = train_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    test_df['X'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)

    # 训练集 验证集划分
    X_train, X_val, y_train, y_val = train_test_split(train_df['X'], train_df['Report'],
                                                      test_size=0.002,  # 8W*0.002
                                                      random_state=42)

    X_train.to_csv(train_x_seg_path, index=None, header=False)
    y_train.to_csv(train_y_seg_path, index=None, header=False)
    X_val.to_csv(val_x_seg_path, index=None, header=False)
    y_val.to_csv(val_y_seg_path, index=None, header=False)

    test_df['X'].to_csv(test_x_seg_path, index=None, header=False)

    # 9. 填充开始结束符号,未知词填充 oov, 长度填充
    # 使用GenSim训练得出的vocab
    vocab = wv_model.wv.vocab

    # 训练集X处理
    # 获取适当的最大长度
    train_x_max_len = get_max_len(train_df['X'])
    test_X_max_len = get_max_len(test_df['X'])
    X_max_len = max(train_x_max_len, test_X_max_len)
    train_df['X'] = train_df['X'].apply(lambda x: pad_proc(x, X_max_len, vocab))

    # 测试集X处理
    # 获取适当的最大长度
    test_df['X'] = test_df['X'].apply(lambda x: pad_proc(x, X_max_len, vocab))

    # 训练集Y处理
    # 获取适当的最大长度
    train_y_max_len = get_max_len(train_df['Report'])
    train_df['Y'] = train_df['Report'].apply(lambda x: pad_proc(x, train_y_max_len, vocab))

    # 10. 保存pad oov处理后的,数据和标签
    train_df['X'].to_csv(train_x_pad_path, index=None, header=False)
    train_df['Y'].to_csv(train_y_pad_path, index=None, header=False)
    test_df['X'].to_csv(test_x_pad_path, index=None, header=False)

    print('train_x_max_len:{} ,train_y_max_len:{}'.format(X_max_len, train_y_max_len))

    # 11. 词向量再次训练
    print('start retrain w2v model')
    wv_model.build_vocab(LineSentence(train_x_pad_path), update=True)
    wv_model.train(LineSentence(train_x_pad_path), epochs=1, total_examples=wv_model.corpus_count)

    print('1/3')
    wv_model.build_vocab(LineSentence(train_y_pad_path), update=True)
    wv_model.train(LineSentence(train_y_pad_path), epochs=1, total_examples=wv_model.corpus_count)

    print('2/3')
    wv_model.build_vocab(LineSentence(test_x_pad_path), update=True)
    wv_model.train(LineSentence(test_x_pad_path), epochs=1, total_examples=wv_model.corpus_count)

    # 保存词向量模型
    wv_model.save(save_wv_model_path)
    print('finish retrain w2v model')
    print('final w2v_model has vocabulary of ', len(wv_model.wv.vocab))

    # 12. 更新vocab
    vocab = {word: index for index, word in enumerate(wv_model.wv.index2word)}
    reverse_vocab = {index: word for index, word in enumerate(wv_model.wv.index2word)}

    # 保存字典
    save_dict(vocab_path, vocab)
    save_dict(reverse_vocab_path, reverse_vocab)

    # 13. 保存词向量矩阵
    embedding_matrix = wv_model.wv.vectors
    np.save(embedding_matrix_path, embedding_matrix)

    # 14. 数据集转换 将词转换成索引  [<START> 方向机 重 ...] -> [32800, 403, 986, 246, 231
    train_ids_x = train_df['X'].apply(lambda x: transform_data(x, vocab))
    train_ids_y = train_df['Y'].apply(lambda x: transform_data(x, vocab))
    test_ids_x = test_df['X'].apply(lambda x: transform_data(x, vocab))

    # 15. 数据转换成numpy数组
    # 将索引列表转换成矩阵 [32800, 403, 986, 246, 231] --> array([[32800,   403,   986 ]]
    train_X = np.array(train_ids_x.tolist())
    train_Y = np.array(train_ids_y.tolist())
    test_X = np.array(test_ids_x.tolist())

    # 保存数据
    np.save(train_x_path, train_X)
    np.save(train_y_path, train_Y)
    np.save(test_x_path, test_X)
    return train_X, train_Y, test_X


def preprocess_sentence(sentence, max_len, vocab):
    """
    单句话预处理
    """
    # 1. 切词处理
    sentence = sentence_proc(sentence)
    # 2. 填充
    sentence = pad_proc(sentence, max_len - 2, vocab)
    # 3. 转换index
    sentence = transform_data(sentence, vocab)
    return np.array([sentence])


def load_dataset(max_enc_len=200, max_dec_len=50):
    """
    :return: 加载处理好的数据集
    """
    train_X = np.load(train_x_path + '.npy')
    train_Y = np.load(train_y_path + '.npy')
    test_X = np.load(test_x_path + '.npy')

    train_X = train_X[:, :max_enc_len]
    train_Y = train_Y[:, :max_dec_len]
    test_X = test_X[:, :max_enc_len]
    return train_X, train_Y, test_X


def load_train_dataset(max_enc_len, max_dec_len):
    """
    :return: 加载处理好的数据集
    """
    train_X = np.load(train_x_path + '.npy')
    train_Y = np.load(train_y_path + '.npy')

    train_X = train_X[:, :max_enc_len]
    train_Y = train_Y[:, :max_dec_len]
    return train_X, train_Y


def load_test_dataset(max_enc_len=200):
    """
    :return: 加载处理好的数据集
    """
    test_X = np.load(test_x_path + '.npy')
    test_X = test_X[:, :max_enc_len]
    return test_X


def get_max_len(data):
    """
    获得合适的最大长度值
    :param data: 待统计的数据  train_df['Question']
    :return: 最大长度值
    """
    # TODO FIX len size bug
    max_lens = data.apply(lambda x: x.count(' ') + 1)
    return int(np.mean(max_lens) + 2 * np.std(max_lens))


def transform_data(sentence, vocab):
    """
    word 2 index
    :param sentence: [word1,word2,word3, ...] ---> [index1,index2,index3 ......]
    :param vocab: 词表
    :return: 转换后的序列
    """
    # 字符串切分成词
    words = sentence.split(' ')
    # 按照vocab的index进行转换         # 遇到未知词就填充unk的索引
    ids = [vocab[word] if word in vocab else vocab['<UNK>'] for word in words]
    return ids


def pad_proc(sentence, max_len, vocab):
    '''
    # 填充字段
    < start > < end > < pad > < unk > max_lens
    '''
    # 0.按空格统计切分出词
    words = sentence.strip().split(' ')
    # 1. 截取规定长度的词数
    words = words[:max_len]
    # 2. 填充< unk > ,判断是否在vocab中, 不在填充 < unk >
    sentence = [word if word in vocab else '<UNK>' for word in words]
    # 3. 填充< start > < end >
    sentence = ['<START>'] + sentence + ['<STOP>']
    # 4. 判断长度，填充　< pad >
    sentence = sentence + ['<PAD>'] * (max_len - len(words))
    return ' '.join(sentence)


def load_stop_words(stop_word_path):
    '''
    加载停用词
    :param stop_word_path:停用词路径
    :return: 停用词表 list
    '''
    # 打开文件
    file = open(stop_word_path, 'r', encoding='utf-8')
    # 读取所有行
    stop_words = file.readlines()
    # 去除每一个停用词前后 空格 换行符
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words


# 加载停用词
stop_words = load_stop_words(stop_word_path)


def clean_sentence(sentence):
    '''
    特殊符号去除
    :param sentence: 待处理的字符串
    :return: 过滤特殊字符后的字符串
    '''
    if isinstance(sentence, str):
        return re.sub(
            # r'[\s+\-\/\[\]\{\}_$%^*(+\"\')]+|[+——()【】“”~@#￥%……&*（）]+|你好|您好',
            r'[\s+\-\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
            ' ', sentence)
    else:
        return ' '


def filter_words(sentence):
    '''
    过滤停用词
    :param seg_list: 切好词的列表 [word1 ,word2 .......]
    :return: 过滤后的停用词
    '''
    words = sentence.split(' ')
    # 去掉多余空字符
    words = [word for word in words if word]
    # 去掉停用词 包括一下标点符号也会去掉
    words = [word for word in words if word not in stop_words]
    return words


def seg_proc(sentence):
    tokens = sentence.split('|')
    result = []
    for t in tokens:
        result.append(cut_sentence(t))
    return ' | '.join(result)


def cut_sentence(line):
    # 切词，默认精确模式，全模式cut参数cut_all=True
    tokens = jieba.cut(line)
    return ' '.join(tokens)


def sentence_proc(sentence):
    '''
    预处理模块
    :param sentence:待处理字符串
    :return: 处理后的字符串
    '''
    # 清除无用词
    sentence = clean_sentence(sentence)
    # 分段切词
    sentence = seg_proc(sentence)
    # 过滤停用词
    words = filter_words(sentence)
    # 拼接成一个字符串,按空格分隔
    return ' '.join(words)


def sentences_proc(df):
    '''
    数据集批量处理方法
    :param df: 数据集
    :return:处理好的数据集
    '''
    # 批量预处理 训练集和测试集
    for col_name in ['Brand', 'Model', 'Question', 'Dialogue']:
        df[col_name] = df[col_name].apply(sentence_proc)

    if 'Report' in df.columns:
        # 训练集 Report 预处理
        df['Report'] = df['Report'].apply(sentence_proc)
    return df


if __name__ == '__main__':
    # 数据集批量处理
    build_dataset(train_data_path, test_data_path)
