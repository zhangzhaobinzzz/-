# -*- coding:utf-8 -*-
import tensorflow as tf
import os


def config_gpu(use_cpu=False):
    if use_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # gpu报错 使用cpu运行
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

                # for gpu in gpus:
                #     tf.config.experimental.set_memory_growth(gpu, True)

            except RuntimeError as e:
                print(e)
