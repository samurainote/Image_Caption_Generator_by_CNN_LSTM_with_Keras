from vgg16 import VGG16
from keras.applications import inception_v3
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
from keras.preprocessing import image, sequence
from keras.callbacks import ModelCheckpoint
import pickle

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

def variable_initializer(self):
        df = pd.read_csv('Flickr8k_text/flickr_8k_train_dataset.txt', delimiter='\t')
        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        for i in range(nb_samples):
            x = iter.next()
            caps.append(x[1][1])

        self.total_samples=0
        for text in caps:
            self.total_samples+=len(text.split())-1
        print "Total samples : "+str(self.total_samples)

        words = [txt.split() for txt in caps]
        unique = []
        for word in words:
            unique.extend(word)

        unique = list(set(unique))
        self.vocab_size = len(unique)
        self.word_index = {}
        self.index_word = {}
        for i, word in enumerate(unique):
            self.word_index[word]=i
            self.index_word[i]=word

        max_len = 0
        for caption in caps:
            if(len(caption.split()) > max_len):
                max_len = len(caption.split())
        self.max_cap_len = max_len
        print "Vocabulary size: "+str(self.vocab_size)
        print "Maximum caption length: "+str(self.max_cap_len)
        print "Variables initialization done!"






        

def load_image(self, path):
        img = image.load_img(path, target_size=(224,224))
        x = image.img_to_array(img)
        return np.asarray(x)

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
