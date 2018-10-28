import numpy as np
from chainer import Chain, Variable
from chainer import functions as F
from chainer import links as L
import yaml
import nltk
import matplotlib.pyplot as plt



class Encoder(Chain):

    def __init__(self, size_vocab, size_embed, size_hidden, size_batch):
        '''
        : arg size_vocab  : 使われる単語の語彙数
        : arg size_embed  : 単語をベクトル表現した時のサイズ
        : arg size_hidden : 隠れ層のサイズ
        : arg size_batch  : バッチのサイズ
        '''
        super(Encoder, self).__init__()
        self.size_batch  = size_batch
        self.size_hidden = size_hidden

        with self.init_scope():
            self.embed = L.EmbedID(size_vocab, size_embed, ignore_label=-1)
            self.lstm  = L.LSTM(size_embed, size_hidden)

    def __call__(self, x):
        """
        : arg x (size_batch)              : 単語ID
        : arg h (size_batch, size_hidden) : 前回の隠れ層
        : ret h (size_batch, size_hidden) : 今回の隠れ層
        """
        e = F.tanh(self.embed(x))
        h = self.lstm(e)
        return h

    def reset(self):
        self.lstm.reset_state()



class Decoder(Chain):

    def __init__(self, size_vocab, size_embed, size_hidden, size_batch):
        '''
        : arg size_vocab  : 使われる単語の語彙数
        : arg size_embed  : 単語をベクトル表現した時のサイズ
        : arg size_hidden : 隠れ層のサイズ
        : arg size_batch  : バッチのサイズ
        '''
        super(Decoder, self).__init__()
        self.size_batch  = size_batch
        self.size_hidden = size_hidden

        with self.init_scope():
            self.embed = L.EmbedID(size_vocab, size_embed, ignore_label=-1)
            self.lstm  = L.LSTM(size_embed, size_hidden)
            #self.he    = L.Linear(size_hidden, size_embed) #
            #self.ev    = L.Linear(size_embed , size_vocab) #
            self.hv = L.Linear(size_hidden, size_vocab)

    def __call__(self, x):
        """
        : arg x (size_batch)              : 単語ID
        : arg h (size_batch, size_hidden) : 前回の隠れ層
        : ret t (size_batch, size_vocab ) : 出力
        : ret h (size_batch, size_hidden) : 今回の隠れ層
        """
        e = F.tanh(self.embed(x))
        h = self.lstm(e)
        #t = self.ev(F.tanh(self.he(h))) #出力は隠れ層を(size_batch, size_vocab)のonehotに
        t = self.hv(h)
        return t

    def reset(self):
        self.lstm.reset_state()



class Seq2Seq(Chain):

    def __init__(self, size_vocab, size_embed, size_hidden, size_batch):
        '''
        : arg size_vocab  : 使われる単語の語彙数
        : arg size_embed  : 単語をベクトル表現した時のサイズ
        : arg size_hidden : 隠れ層のサイズ
        : arg size_batch  : バッチのサイズ
        '''
        super(Seq2Seq, self).__init__()
        self.size_batch  = size_batch
        self.size_hidden = size_hidden

        with self.init_scope():
            self.encoder = Encoder(size_vocab, size_embed, size_hidden, size_batch)
            self.decoder = Decoder(size_vocab, size_embed, size_hidden, size_batch)

    def __call__(self, enc_b, dec_b):
        '''
        : arg enc_b (size_batch) : encode用の単語のバッチ
        : arg dec_b (size_batch) : decode用の単語のバッチ
        : ret loss               : 計算した損失の合計
        '''
        # model内で使用するLSTMの内部状態をリセット
        self.reset()

        loss = Variable(np.zeros((), dtype='float32'))

        # encode
        for w in enc_b:
            h = self.encoder(w)

        # encoderの隠れ層の値をdecorderに受け渡し
        self.decoder.lstm.h = h

        # decord
        for w, w_ in zip(dec_b, dec_b[1:]):
            t = self.decoder(w)
            loss += F.softmax_cross_entropy(t, w_) #ラベル w_ はIDのままでOK

        # 学習前に内部の勾配をリセット
        self.cleargrads()

        return loss

    #
    def reset(self):
        """
        中間ベクトル、内部メモリ、勾配の初期化
        """
        self.encoder.reset()
        self.decoder.reset()
