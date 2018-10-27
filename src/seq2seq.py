import numpy as np
from chainer import Chain, Variable, optimizer, optimizers, serializers
from chainer import functions as F
from chainer import links as L
import yaml, pickle
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
            #self.upward  = L.Linear(size_embed , 4 * size_hidden)
            #self.lateral = L.Linear(size_hidden, 4 * size_hidden)
            self.lstm  = L.LSTM(size_embed, size_hidden)

    def __call__(self, x):
        """
        : arg x (size_batch)              : 単語ID
        : arg h (size_batch, size_hidden) : 前回の隠れ層
        : ret h (size_batch, size_hidden) : 今回の隠れ層
        """
        e = F.tanh(self.embed(x))
        #self.c, h = F.lstm(self.c, self.upward(e) + self.lateral(h))
        h = self.lstm(e)

        return h

    def reset(self):
        #self.c = np.zeros((self.size_batch, self.size_hidden), dtype='float32')
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
            #self.upward  = L.Linear(size_embed , size_hidden * 4) #LSTM内部の再現用
            #self.lateral = L.Linear(size_hidden, size_hidden * 4) #LSTM内部の再現用
            self.lstm  = L.LSTM(size_embed, size_hidden)
            self.he    = L.Linear(size_hidden, size_embed)
            self.ev    = L.Linear(size_embed , size_vocab)

    def __call__(self, x):
        """
        : arg x (size_batch)              : 単語ID
        : arg h (size_batch, size_hidden) : 前回の隠れ層
        : ret t (size_batch, size_vocab ) : 出力
        : ret h (size_batch, size_hidden) : 今回の隠れ層
        """
        e = F.tanh(self.embed(x))
        #self.c, h = F.lstm(self.c, self.upward(e) + self.lateral(h))
        h = self.lstm(e)
        t = self.ev(F.tanh(self.he(h)))

        return t

    def reset(self):
        #self.c = np.zeros((self.size_batch, self.size_hidden), dtype='float32')
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
        : arg enc_b (size_batch):
        : arg dec_b (size_batch) :
        : ret loss               : 計算した損失の合計
        '''
        # model内で使用するLSTMの内部状態をリセット
        model.reset()

        loss = Variable(np.zeros((), dtype='float32'))

        # encode
        for w in enc_b:
            h = model.encoder(w)

        # encoderの隠れ層の値をdecorderに受け渡し
        model.decoder.lstm.h = h

        # decord
        for w, w_ in zip(dec_b, dec_b[1:]):
            t = model.decoder(w)
            loss += F.softmax_cross_entropy(t, w_) #ラベル w_ はIDのままでOK

        model.cleargrads()

        return loss

    #
    def reset(self):
        """
        中間ベクトル、内部メモリ、勾配の初期化
        """
        self.encoder.reset()
        self.decoder.reset()



# main
if __name__ == '__main__':
    # データのパラメータ取得
    with open('config.yml', 'r+') as f:
        config = yaml.load(f)

    with open(config['path']['index_dict'], 'rb') as f:
        index_dict = pickle.load(f)
    with open(config['path']['batched'], 'rb') as f:
        batches = pickle.load(f)

    n_epoch     = config['param']['n_epoch']
    n_vocab     = len(index_dict)
    size_embed  = config['param']['size_embed']
    size_hidden = config['param']['size_hidden']
    size_batch  = config['param']['size_batch']


    # 学習の設定
    model = Seq2Seq(n_vocab, size_embed, size_hidden, size_batch)
    opt = optimizers.Adam()
    opt.setup(model)
    opt.add_hook(optimizer.GradientClipping(5))


    # 学習開始
    loss_list = []
    for epoch in range(n_epoch):
        sum_loss = 0
        for enc_b, dec_b in batches:
            loss = model(enc_b, dec_b)
            model.reset()
            loss.backward()
            opt.update()

            sum_loss += loss.data
        print('{:3} | {}'.format(epoch+1, sum_loss))
        loss_list.append(sum_loss)

    '''
    plt.plot(loss_list)
    plt.savefig(config[''])
    serializers.save_npz('seq2seq_100.npz', model)
    '''
