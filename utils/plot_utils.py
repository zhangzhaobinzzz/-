# -*- coding:utf-8 -*-

from matplotlib import font_manager

import matplotlib.ticker as ticker

# 解决中文乱码
font = font_manager.FontProperties(fname="data/TrueType/simhei.ttf")
import matplotlib.pyplot as plt


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 12, 'fontproperties': font}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
