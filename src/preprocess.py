'''
Seq2Seqに使用するテキストデータの前処理

onehot処理:
  - 単語をonehotベクトルに変換する辞書の作成
  - 各単語をonehot化
バッチ処理:
  - バッチに分割
  - decテキストの先頭に<START>を追加
  - enc,decテキストの両方の末尾に<END>を追加
  - バッチごとに文章の長さを揃える(短いものは<PAD>で埋める)
'''

import numpy as np
import nltk
import yaml, pickle


''' データの準備 '''
with open('config.yml', 'r+') as f: config = yaml.load(f)

with open(config['path']['raw_enc']) as f: enc_texts = f.readlines()
with open(config['path']['raw_dec']) as f: dec_texts = f.readlines()
# 文章を単語のリストに変換
enc_texts = [nltk.word_tokenize(t) for t in enc_texts]
dec_texts = [nltk.word_tokenize(t) for t in dec_texts]


''' index処理 '''
index_dict = {'<START>':0, '<END>':1}
for t in enc_texts + dec_texts:
    for w in t:
        if w not in index_dict: index_dict[w] = len(index_dict)
#print(index_dict)
# インデックス化
enc_texts = [[index_dict[w] for w in t] for t in enc_texts]
dec_texts = [[index_dict[w] for w in t] for t in dec_texts]

''' バッチ処理 '''
size_batch = config['param']['size_batch']
# バッチに分割
batches = []
for i in range(0, min(len(enc_texts), len(dec_texts)), size_batch):
    # encorderのバッチ生成
    enc_b = enc_texts[i:i+size_batch]
    max_len = max(len(t) for t in enc_b)                 #バッチ内での文の長さの最長
    enc_b = [t + [-1] * (max_len-len(t)) for t in enc_b] #最長の文に長さ合わせて<PAD>埋め
    enc_b = np.array(enc_b, dtype='int32').T              #転置
    # decorderのバッチ生成
    dec_b = dec_texts[i:i+size_batch]
    max_len = max(len(t) for t in dec_b)                 #バッチ内での文の長さの最長
    dec_b = [t + [-1] * (max_len-len(t)) for t in dec_b] #最長の文に長さ合わせて<PAD>埋め
    dec_b = np.array(dec_b, dtype='int32').T              #転置
    # 1バッチはencorderとdecorderのタプル
    batches.append((enc_b, dec_b))


''' 保存 '''
with open(config['path']['index_dict'], 'wb') as f:
    pickle.dump(index_dict, f)
with open(config['path']['batched'], 'wb') as f:
    pickle.dump(batches, f)
