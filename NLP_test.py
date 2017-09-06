# coding: utf-8

# # NLP test

import sqlite3
import matplotlib.pyplot as plt
import pickle
import sklearn
import numpy as np
import pandas as pd
import collections
import sys
from datetime import datetime

def vec2word(word_vec):
    return word_model.most_similar(positive=[word_vec], topn=1)


def get_obj():
    # obj 生成
    conn = sqlite3.connect('reviews.sqlite3')
    cur = conn.cursor()

    cur.execute("select content from spams")
    obj = cur.fetchall()

    conn.close()

    return obj

def get_all_data(obj, os_type="mac", sentence_num=5000):
    '''
    形態素解析Mecab + neologd版
    all_dataはsentenceごとのひらがな分かち書き
    whitelistの単語のみ許容
    sentece末尾に終端文字を入れる
    '''
    import MeCab
    import jaconv

    if os_type == "mac":
        mecab = MeCab.Tagger ('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')   #for mac
    elif os_type == "ubuntu":
        mecab = MeCab.Tagger ('-d /usr/lib/mecab/dic/mecab-ipadic-neologd') # for ubuntu

    all_data = []
    for art in obj[:sentence_num]:
        art = art[0]

        mecab.parse("")
        node = mecab.parseToNode(art)

        tmp_data = []
        while node:
            #単語を取得
            word = node.surface
            #品詞を取得
            pos = node.feature.split(",")

            # 記号とwhitelist以外の単語を弾く
            if pos[0] != '記号':
                word = jaconv.kata2hira(pos[-2])
                if (set(word) - whitelist) == set():
                    tmp_data.append(jaconv.kata2hira(pos[-2]))

            #次の単語に進める
            node = node.next

        # 終端文字挿入
        tmp_data.append("E")

        all_data.append(tmp_data)

    return all_data


if __name__ == "__main__":
    os_type = sys.argv[1]

    whitelist = "\n「」。、あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゐゆゑよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉゃゅょっゎ"
    whitelist = set(whitelist)
    print("whitelist:",whitelist)

    obj = get_obj()

    all_data = get_all_data(obj,os_type=sys.argv[1])

    '''
    全単語の辞書とword2vec作成
    '''
    all_word = []
    for s in all_data:
        all_word.extend(s)

    # 全単語ベクトル
    word_id_dict = {k:v for v,k in enumerate(list(set(all_word)))}
    id_word_dict = {k:v for v,k in word_id_dict.items()}
    word_num = len(word_id_dict)

    with open("corpus.txt","w") as f:
        for i in all_word:
            f.write(i+" ")

    # word2vec
    print("Build word2vec")
    from gensim.models import word2vec

    sentences = word2vec.Text8Corpus('corpus.txt')

    word_model = word2vec.Word2Vec(sentences,
                              sg=1,
                              size=200,
                              min_count=1,
                              window=10,
                              hs=1,
                              negative=0)

    # sentenceごと，trigramのx_dataと次の単語のy_data生成
    x_data = []
    y_data = []
    for s in all_data:
        s_len = len(s)
        if s_len < 4:
            continue
        trigram = []
        next_word = []
        for i in range(s_len - 3):
            trigram.append(s[i:i+3])
            next_word.append(word_id_dict[s[i+3]])

        x_data.extend([[word_model[y] for y in x] for x in trigram])
        y_data.extend(next_word)
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    print(np.array(x_data).shape)
    print(np.array(y_data).shape)
    print(len(word_id_dict))

    # one hot vectorize
    print("one hot vectorize")
    x_train = x_data

    y_train = y_data

    from keras.utils import np_utils

    nb_classes = len(word_id_dict)

    Y_train = np_utils.to_categorical(y_train, nb_classes)

    print("x_train shape:",x_train.shape)
    print("Y_train shape:",Y_train.shape)

    # LSTM model
    print("Build LSTM model")
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.layers import LSTM
    from keras.optimizers import RMSprop

    model = Sequential()
    model.inputs
    model.add(LSTM(128, batch_input_shape=(1,3,200), stateful=True))
    model.add(Dense(word_num))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    print("Fitting")
    history = model.fit(x_train, Y_train, batch_size=1, epochs=1, shuffle=False)

    import h5py
    fin_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    model.save(fin_time+".h5")
    print('save model:',fin_time+".h5")

    # 学習の様子をプロット
    loss = history.history['loss']
    # val_loss = history.history['val_loss']

    acc = history.history['acc']
    # val_acc = history.history['val_acc']

    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    plt.title('Loss')
    epochs = len(loss)
    plt.plot(range(epochs), loss, marker='.', label='loss')
    # plt.plot(range(epochs), val_loss, marker='.', label='val_loss')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('acc')

    plt.subplot(2,1,2)
    plt.title('Accuracy')
    plt.plot(range(epochs), acc, marker='.', label='acc')
    # plt.plot(range(epochs), val_acc, marker='.', label='val_acc')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('acc')

    plt.savefig(fin_time+".png")
    print("save fig:",fin_time+".png")
