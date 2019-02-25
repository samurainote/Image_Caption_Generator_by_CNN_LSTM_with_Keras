
from vgg16 import VGG16
from keras.applications import inception_v3
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
from keras.preprocessing import image, sequence
from keras.callbacks import ModelCheckpoint
import pickle

IMAGE_EMBEDDING_DIM = 128

"""
TITLE: Surpervised Image Captioning
GOAL:
INPUT: Image Data
OUTPUT: Sentence which explain given image

ARCHITECTURE: CNN-LSTM Model
1. LSTM

"""

誤差逆伝播 (Back-prop) とは，損失関数を各パラメータで微分して，各パラメータ (Data) における勾配 (Grad) を求め，損失関数が小さくなる方向へパラメータ更新を行うことをいう．
ここで勾配は各パラメータに付随する変数と捉えられる．
Chainer 内部でも，Variable インスタンスにパラメータ (重み行列とバイアスベクトル) を保持する Data と，各パラメータの勾配を保持する Grad の2つがある.
それらに forward メソッド や backward メソッドを適応して Variable を更新している．

1. 入力文 (one-hot ベクトルの配列) を埋め込み層の行列で埋め込み，エンコーダ LSTM で内部状態 S (最終タイムステップの LSTM 内の cell state と hidden state) に縮約する．
2. S をデコーダ LSTM の内部状態にセットし，SOS (Start of Sentence) の埋め込みベクトルをデコーダ LSTM に入力し，予測単語ベクトルを出力する．
3. 予測単語ベクトルとソフトマックス層の行列 (単語ベクトルの配列) の内積から確率分布を生成し，最大確率に対応する単語を出力する．
4. 出力した単語を埋め込み，次のタイムステップのデコーダ LSTM に入力し，予測単語ベクトルを出力する．
5. デコーダ LSTM が EOS (End of Sentence) を出力するまでステップ 3 と 4 を繰り返す．

"""
LSTMというのは基本的に横軸の流れ時間軸を表す
重ねることで縦軸も生まれる
EnceodrとDecoderのなかに打ち込むこと

白色のセルは LSTM の各タイムステップ

LSTM は「情報をどの程度通すか」をゲーティングによって決める．
入力ゲートは通常入力を更新後の cell state に加える程度を決めるゲーティング，
忘却ゲートは更新前の cell state を更新後の cell state に残す程度を決めるゲーティング，
出力ゲートは更新後の cell state を hidden state に出力する程度を決めるゲーティング


ct=cell state:長期記憶 (入力 xt が直接 cell に入らずゲート越し) を担当する
mt=hidden state:短期記憶 (入力 xt と豪快に混ざり合う) を担当する
cell state は加算によって状態を更新を行うので勾配消失を回避できる．

cell state: 長期記憶
hidden state: 短期記憶 (正確にはワーキングメモリ)
入力のつど内部状態を更新するモデルである

マトリックス
W  は学習パラメータの重み行列

2つのベクトル
時系列ベクトル：xt は時刻 t の埋め込みベクトル (もしくは下層の隠れ層ベクトルの出力)
出力ベクトル：mt (ht とも表記) は時刻 t−1 の隠れ層 (更新後は時刻 t の隠れ層) の出力ベクトル

4つのゲート
i′t  は RNN と同様の通常入力
it 入力ゲート (Input Gate)
ft 忘却ゲート (Forget Gate)
ot 出力ゲート (Output Gate)
"""


class CaptionGenerator():

    def __init__(self):
        self.MAX_CAP_LEN = None
        self.VOCAB_SIZE = None
        self.IDX2WORD = None
        self.WORD2IDX = None
        self.NUM_SAMPLES = None
        self.ENCODED_IMAGES = pickle.load(open( "encoded_images.p", "rb" ))
        self.VARIABLE_INITIALIZER()

    def vocabulary_creator()

    # def variable_initializer(self):
        df = pd.read_csv('Flickr8k_text/flickr_8k_train_dataset.txt', delimiter='\t')
        num_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        for i in range(num_samples):
            x = iter.next()
            caps.append(x[1][1])



        self.NUM_SAMPLES=0
        for text in caps:
            self.NUM_SAMPLES+=len(text.split())-1
        print "Total samples : "+str(self.NUM_SAMPLES)

        words = [text.split() for text in caps]
        unique_words = []
        for word in words:
            unique_words.extend(word)

        unique_words = list(set(unique_words))
        self.VOCAB_SIZE = len(unique_words)
        self.WORD2IDX = {}
        self.IDX2WORD = {}
        for i, word in enumerate(unique_words):
            self.WORD2IDX[word] = i
            self.IDX2WORD[i] = word

        max_len = 0
        for caption in caps:
            if len(caption.split()) > max_len:
                max_len = len(caption.split())
        self.MAX_CAP_LEN = max_len
        print "Vocabulary size: "+str(self.VOCAB_SIZE)
        print "Maximum caption length: "+str(self.MAX_CAP_LEN)
        print "Variables initialization done!"


    def data_generator(self, batch_size = 32):
        partial_caps = []
        next_words = []
        images = []
        print "Generating data..."
        gen_count = 0
        df = pd.read_csv('Flickr8k_text/flickr_8k_train_dataset.txt', delimiter='\t')
        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        imgs = []
        for i in range(nb_samples):
            x = iter.next()
            caps.append(x[1][1])
            imgs.append(x[1][0])

"""
1. 入力層と出力層のテンソル定義
単語数
説明文の長さ
インデックス-単語辞書
単語辞書-インデックス
"""


        total_count = 0
        while 1:
            image_counter = -1
            for text in caps:
                image_counter+=1
                current_image = self.encoded_images[imgs[image_counter]]
                for i in range(len(text.split())-1):
                    total_count+=1
                    partial = [self.word_index[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)
                    next = np.zeros(self.vocab_size)
                    next[self.word_index[text.split()[i+1]]] = 1
                    next_words.append(next)
                    images.append(current_image)

                    if total_count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=self.max_cap_len, padding='post')
                        total_count = 0
                        gen_count+=1
                        print "yielding count: "+str(gen_count)
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []

    def load_image(self, path):
        img = image.load_img(path, target_size=(224,224))
        x = image.img_to_array(img)
        return np.asarray(x)


    def create_model(self, ret_model = False):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
        base_model.trainable=False

        image_model = Sequential()
        image_model.add(base_model)
        image_model.add(Flatten())
        image_model.add(Dense(EMBEDDING_DIM, input_dim = 4096, activation='relu'))

        image_model.add(RepeatVector(self.max_cap_len))

        lang_model = Sequential()
        lang_model.add(Embedding(self.vocab_size, 256, input_length=self.max_cap_len))
        lang_model.add(LSTM(256,return_sequences=True))
        lang_model.add(TimeDistributed(Dense(EMBEDDING_DIM)))

        model = Sequential()
        model.add(Merge([image_model, lang_model], mode='concat'))
        model.add(LSTM(1000,return_sequences=False))
        model.add(Dense(self.vocab_size))
        model.add(Activation('softmax'))

def create_model(self, ret_model = False):


        image_model = Sequential()
        # units：正の整数，出力空間の次元数
        image_model.add(Dense(units=IMAGE_EMBEDDING_DIM, input_dim=4096, activation='relu'))
        image_model.add(RepeatVector(n=self.MAX_CAP_LEN)) # Denseから得られる固定長ベクトルを出力の長さ分だけ繰り返し

        lang_model = Sequential()
        lang_model.add(Embedding(input_dim=self.VOCAB_SIZE, output_dim=256, input_length=self.max_cap_len))
        lang_model.add(LSTM(256,return_sequences=True))
        lang_model.add(TimeDistributed(Dense(IMAGE_EMBEDDING_DIM)))

        model = Sequential()
        model.add(Merge([image_model, lang_model], mode='concat'))
        model.add(LSTM(1000,return_sequences=False))
        model.add(Dense(self.vocab_size))
        model.add(Activation('softmax'))


入力層 Numeric 辞書化
自然言語
id

Embedding 埋め込み
単語数（文章数）
意味ベクトルの次元数

LSTM
意味ベクトルの次元数
256次元

Dense + ソフトマックス 全結合層
256次元
出力空間の次元数（辞書サイズ・単語数）
softmax 確率分布

出力層
softmax 確率分布
文字化！



ラッパー RepeatVector

RepeatVectorは、inputとして入ってくるベクトルを、指定した数だけ繰り返すラッパーです。
Encoderから得られる固定長ベクトルを出力の長さ分だけ繰り返して、毎時刻入力できるようにしています。


Encoderネットワーク
シーケンス入力: [A, B, C, (EOS)]
固定長ベクトル:

Decoderネットワーク
固定長ベクトル:
シーケンス出力: [W, X, Y, Z, (EOS)]

I am Sam.
文の長さ 3
文の意味埋め込みベクトルの次元 100

モデル
長さ7のシーケンス
各要素が12次元のベクトル
128次元のRNN層
全結合層で12次元にまとめる
ソフトマックスで活性化する
入力シーケンスを12クラスに分類する

RNNのインプット
IN:7 * 12　の２次元テンソル（行列）
OUT:128次元のRNNベクトル

全結合層
IN:128次元のRNNベクトル
OUT:12次元にまとめる

seq_length = 7
n_in = 12
n_hidden = 128
n_out = 12

model=Sequential()
model.add(SimpleRNN(units=n_hidden=128, input_shape=(seq_length=7, n_in=12)))
model.add(Dense(units=n_out=12))
model.add(Activation('softmax'))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss='categorical_cross_entropy', optimizer=optimizer)


ADAM = Adam(lr=0.00005)

"""
Input Layer #Document*2
"""
input_context = Input(shape=(MAX_INPUT_LENGTH,), dtype="int32", name="input_context")
input_answer = Input(shape=(MAX_INPUT_LENGTH,), dtype="int32", name="input_answer")

"""
Embedding Layer: 正の整数（インデックス）を固定次元の密ベクトルに変換します．
・input_dim: 正の整数．語彙数．入力データの最大インデックス + 1．
・output_dim: 0以上の整数．密なembeddingsの次元数．
・input_length: 入力の系列長（定数）． この引数はこのレイヤーの後にFlattenからDenseレイヤーへ接続する際に必要です (これがないと，denseの出力のshapeを計算できません)．
"""
# weightが存在したら引用する
if os.path.isfile(weights_file):
    Shared_Embedding = Embedding(input_dim=DICTIONARY_SIZE, output_dim=WORD2VEC_DIMS, input_length=MAX_INPUT_LENGTH,)
else:
    Shared_Embedding = Embedding(input_dim=DICTIONARY_SIZE, output_dim=WORD2VEC_DIMS, input_length=MAX_INPUT_LENGTH,
                                 weights=[word_embedding_matrix])

"""
Shared Embedding Layer #Doc2Vec(Document*2)
"""
shared_embedding_context = Shared_Embedding(input_context)
shared_embedding_answer = Shared_Embedding(input_answer)

"""
LSTM Layer #
"""
Encoder_LSTM = LSTM(units=DOC2VEC_DIMS, init= "lecun_uniform")
Decoder_LSTM = LSTM(units=DOC2VEC_DIMS, init= "lecun_uniform")
embedding_context = Encoder_LSTM(shared_embedding_context)
embedding_answer = Decoder_LSTM(shared_embedding_answer)

"""
Merge Layer #
"""
merge_layer = merge([embedding_context, embedding_answer], mode='concat', concat_axis=1)

"""
Dense Layer #
"""
dence_layer = Dense(DICTIONARY_SIZE/2, activation="relu")(merge_layer)

"""
Output Layer #
"""
outputs = Dense(DICTIONARY_SIZE, activation="softmax")(dence_layer)

"""
Modeling
"""
model = Model(input=[input_context, input_answer], output=[outputs])
model.compile(loss="categorical_crossentropy", optimizer=ADAM)
