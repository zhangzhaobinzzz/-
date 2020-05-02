# -*- coding:utf-8 -*-
import tensorflow as tf

from seq2seq_tf2.model_layers import Encoder, BahdanauAttention, Decoder
from utils.gpu_utils import config_gpu
from utils.params_utils import get_params
from utils.wv_loader import load_embedding_matrix, Vocab


class Seq2Seq(tf.keras.Model):
    def __init__(self, params):
        super(Seq2Seq, self).__init__()
        self.embedding_matrix = load_embedding_matrix()
        self.params = params
        self.encoder = Encoder(params["vocab_size"],
                               params["embed_size"],
                               self.embedding_matrix,
                               params["enc_units"],
                               params["batch_size"])

        self.attention = BahdanauAttention(params["attn_units"])

        self.decoder = Decoder(params["vocab_size"],
                               params["embed_size"],
                               self.embedding_matrix,
                               params["dec_units"],
                               params["batch_size"])

    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
        return enc_output, enc_hidden

    def call_decoder_onestep(self, dec_input, dec_hidden, enc_output):
        context_vector, attention_weights = self.attention(dec_hidden, enc_output)

        pred, dec_hidden = self.decoder(dec_input,
                                        None,
                                        None,
                                        context_vector)
        return pred, dec_hidden, context_vector, attention_weights

    def call(self, dec_input, dec_hidden, enc_output, dec_target):
        predictions = []
        attentions = []

        context_vector, _ = self.attention(dec_hidden, enc_output)

        for t in range(1, dec_target.shape[1]):
            pred, dec_hidden = self.decoder(dec_input,
                                            dec_hidden,
                                            enc_output,
                                            context_vector)

            context_vector, attn = self.attention(dec_hidden, enc_output)
            # using teacher forcing
            dec_input = tf.expand_dims(dec_target[:, t], 1)

            predictions.append(pred)
            attentions.append(attn)

        return tf.stack(predictions, 1), dec_hidden


if __name__ == '__main__':
    # GPU资源配置
    config_gpu()
    # 获得参数
    params = get_params()
    # 读取vocab训练
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    # 计算vocab size
    vocab_size = vocab.count
    batch_size = 128
    input_sequence_len = 200

    params = {}
    params["vocab_size"] = vocab_size
    params["embed_size"] = 500
    params["enc_units"] = 512
    params["attn_units"] = 512
    params["dec_units"] = 512
    params["batch_size"] = batch_size

    model = Seq2Seq(params)

    # example_input
    example_input_batch = tf.ones(shape=(batch_size, input_sequence_len), dtype=tf.int32)

    # sample input
    sample_hidden = model.encoder.initialize_hidden_state()

    sample_output, sample_hidden = model.encoder(example_input_batch, sample_hidden)
    # 打印结果
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

    attention_layer = BahdanauAttention(10)
    context_vector, attention_weights = attention_layer(sample_hidden, sample_output)

    print("Attention context_vector shape: (batch size, units) {}".format(context_vector.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    sample_decoder_output, _, = model.decoder(tf.random.uniform((batch_size, 1)),
                                              sample_hidden, sample_output, context_vector)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
