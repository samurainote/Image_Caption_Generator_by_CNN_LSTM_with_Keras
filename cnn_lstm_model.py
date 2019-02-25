from vgg16 import VGG16
from keras.applications import inception_v3
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer, image, sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Input, LSTM, Embedding, TimeDistributed, Dense, Dropout, RepeatVector, Merge, Activation, Flatten, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.utils import plot_model
import pickle
from pickle import load


IMG_PIXEL = 224
IMG_SAMPLE_SIZE = 4096
IMG_EMBEDDING_DIM = 128

MAX_CAP_LEN = 30
VOCAB_SIZE = 10000
CAP_SAMPLE_SIZE =
CAP_EMBEDDING_DIM = 128

index_word_dict =
word_index_dict =

# LSTM_LAYERS = 2
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001


# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(desc) for desc in descriptions[key]]
    return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# calculate the length of the description with the most words
def max_cap_len(descriptions):
    lines = to_lines(descriptions)
    return max(len(desc.split()) for desc in lines)

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo):
    vocab_size = len(tokenizer.word_index) + 1

    X1, X2, y = [], [], []
    # walk through each description for the image
    for desc in desc_list:
        # split one sequence into multiple X,y pairs
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)


# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length, n_step = 1):
    # loop for ever over images
    while 1:
      # loop over photo identifiers in the dataset
      keys = list(descriptions.keys())
      for i in range(0, len(keys), n_step):
        Ximages, XSeq, y = list(), list(),list()
        for j in range(i, min(len(keys), i+n_step)):
          image_id = keys[j]
          # retrieve the photo feature
          photo = photos[image_id][0]
          desc_list = descriptions[image_id]
          in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
          for k in range(len(in_img)):
            Ximages.append(in_img[k])
            XSeq.append(in_seq[k])
            y.append(out_word[k])
        yield [[np.array(Ximages), np.array(XSeq)], np.array(y)]

def categorical_crossentropy_from_logits(y_true, y_pred):
    y_true = y_true[:, :-1, :]  # Discard the last timestep
    y_pred = y_pred[:, :-1, :]  # Discard the last timestep
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return loss

def categorical_accuracy_with_variable_timestep(y_true, y_pred):
    y_true = y_true[:, :-1, :]  # Discard the last timestep
    y_pred = y_pred[:, :-1, :]  # Discard the last timestep
    # Flatten the timestep dimension
    shape = tf.shape(y_true)
    y_true = tf.reshape(y_true, [-1, shape[-1]])
    y_pred = tf.reshape(y_pred, [-1, shape[-1]])

    # Discard rows that are all zeros as they represent padding words.
    is_zero_y_true = tf.equal(y_true, 0)
    is_zero_row_y_true = tf.reduce_all(is_zero_y_true, axis=-1)
    y_true = tf.boolean_mask(y_true, ~is_zero_row_y_true)
    y_pred = tf.boolean_mask(y_pred, ~is_zero_row_y_true)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=1),
                                                tf.argmax(y_pred, axis=1)),
                                      dtype=tf.float32))
    return accuracy

def Glove_word2vectorizer():

    embeddings_index = dict()
    f = open('glove.6B/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((vocabulary_size, 100))
    for word, index in tokenizer.word_index.items():
        if index > vocabulary_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    return embedding_matrix

# Transfer Learning
def cnn_lstm_model_creater(img_pixel_len, img_data_size, img_embedding_dim,
                           cap_pixel_len, cap_vocab_size, cap_embedding_dim, ):
    #base_model = VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
    img_encoder_inputs = Input(shape=(IMG_PIXEL, IMG_PIXEL, 3), dtype="float32")
    img_pretrained_model = VGG16(weights="imagenet", include_top=False, input_tensor=img_encoder_inputs).trainable=False
    img_encoder_dropout = Dropout(rate=DROPOUT_RATE)(img_pretrained_model)
    img_dense_layer = Dense(units=IMG_EMBEDDING_DIM, input_dim=IMG_SAMPLE_SIZE, activation='relu')(img_encoder_dropout)
    img_repeat_vector = RepeatVector(n=MAX_CAP_LEN)(img_dense_layer)

    # word2vec = Glove
    cap_encoder_inputs = Input(shape=(MAX_CAP_LEN, ), dtype="float32")
    cap_encoder_embedding = Embedding(output_dim=CAP_EMBEDDING_DIM, input_dim=VOCAB_SIZE, input_length=MAX_CAP_LEN,
                                      embeddings_initializer=Constant(embedding_matrix), trainable=False))(cap_encoder_inputs)
    cap_encoder_lstm = LSTM(units=CAP_EMBEDDING_DIM, return_sequences=True)(cap_encoder_embedding)
    cap_encoder_time = TimeDistributed(Dense(units=CAP_EMBEDDING_DIM))(cap_encoder_lstm)

    # merge_layer
    merge_layer = concatenate([img_repeat_vector, cap_encoder_time])
    # lstm_layers
    decoder_lstm = LSTM(units=CAP_EMBEDDING_DIM, return_sequences=True)(merge_layer)
    sampling_layer = Dense(units=VOCAB_SIZE, activation="softmax")(decoder_lstm)

    # compile it!
    model = Model(inputs=[img_encoder_inputs, cap_encoder_inputs], outputs=sampling_layer)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    print(model.summary())
    plot_model(model, show_shapes=True, to_file='_model.png')

    return model


"""
Input: 1データの形状 Vectorベクトル = 文章の長さ
Embedding: データの数と1データの形状　MATRIX行列 = (辞書サイズ, 文章の長さ)

import gensim

# Load Google's pre-trained Word2Vec model.
model = gensim.models.Word2Vec.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
# Glove
"""


"""
CNN
画像データ
ピクセルサイズ（x*y）
ウィンドウサイズ

辞書: vocabulary_creator
入力層作り: (VOCAB_SIZE*MAX_CAP_LEN) + IDX2WORDで単語を数値に
出力層づくり: (VOCAB_SIZE*MAX_CAP_LEN) + WORD2IDX(argmax)で単語出力

Input: センテンス->id
Embedding: id->意味ベクトル
LSTM: 意味ベクトル->時系列配置

Input: 画像->
Embedding: id->意味ベクトル
LSTM: 意味ベクトル->時系列配置
"""
