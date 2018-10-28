import numpy as np
from chainer import datasets, iterators, optimizer, optimizers, serializers
import seq2seq
import yaml
import nltk
import os, shutil
import matplotlib.pyplot as plt



def arrange_seqBatch(batch, pad=-1):
    max_len = max(len(d) for d in batch)                 #バッチ内での文の長さの最長
    return [d + [pad] * (max_len-len(d)) for d in batch] #最長の文に長さ合わせて<PAD>埋め



if __name__ == '__main__':

    ''' 設定ファイルの読みこみ '''
    with open('config.yml', 'r+') as f: config = yaml.load(f)
    n_epoch     = config['param']['n_epoch']
    size_embed  = config['param']['size_embed']
    size_hidden = config['param']['size_hidden']
    size_batch  = config['param']['size_batch']


    ''' データの準備 '''
    with open(config['path']['raw_enc']) as f: enc_texts = f.readlines()
    with open(config['path']['raw_dec']) as f: dec_texts = f.readlines()

    print(len(enc_texts))
    # 文章を単語のリストに変換
    enc_texts = [nltk.word_tokenize(t) for t in enc_texts]
    dec_texts = [nltk.word_tokenize(t) for t in dec_texts]
    # 単語をインデックスに変換する辞書の作成
    index_dict = {'<PAD>':-1, '<START>':0, '<END>':1}
    for t in enc_texts + dec_texts:
        for w in t:
            if w not in index_dict: index_dict[w] = len(index_dict)
    # インデックス化
    enc_texts = [    [index_dict[w] for w in t]+[1] for t in enc_texts]
    dec_texts = [[0]+[index_dict[w] for w in t]+[1] for t in dec_texts]
    # 単語の種類数
    n_vocab = len(index_dict)
    print(n_vocab)


    ''' 学習 '''
    # 学習の設定
    model = seq2seq.Seq2Seq(n_vocab, size_embed, size_hidden, size_batch)
    opt = optimizers.Adam()
    opt.setup(model)
    opt.add_hook(optimizer.GradientClipping(5))

    # 学習開始
    loss_list = []
    for epoch in range(n_epoch):

        sum_loss = 0
        data = list(zip(enc_texts, dec_texts))
        iter = iterators.SerialIterator(data, size_batch, repeat=False, shuffle=True)
        for b in iter:
            # バッチ毎のデータの整形
            enc_b = [e for e, _ in b]
            dec_b = [d for _, d in b]
            enc_b = arrange_seqBatch(enc_b, pad=-1)
            dec_b = arrange_seqBatch(dec_b, pad=-1)
            enc_b = np.array(enc_b, dtype='int32').T
            dec_b = np.array(dec_b, dtype='int32').T
            # 学習
            loss = model(enc_b, dec_b)
            loss.backward()
            opt.update()
            sum_loss += loss.data

        print('{:3} | {}'.format(epoch+1, sum_loss))
        loss_list.append(sum_loss)


    ''' 結果の保存 '''
    dir = config['path']['out_dir']

    os.makedirs(dir, exist_ok=True)
    shutil.copy('config.yml', dir+'/config.yml')

    plt.plot(loss_list)
    plt.savefig(dir+'/loss.jpg')
    plt.show()

    serializers.save_npz(dir+'/seq2seq.npz', model)
