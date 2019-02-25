import load_data as ld
import cnn_lstm_model as MODEL
from keras.callbacks import ModelCheckpoint
from pickle import dump


IMG_PIXEL = 224
IMG_SAMPLE_SIZE = 4096
IMG_EMBEDDING_DIM = 128

MAX_CAP_LEN = 30
VOCAB_SIZE = 10000
CAP_SAMPLE_SIZE =
CAP_EMBEDDING_DIM = 128

NUM_EPOCHS = 10

def train_model(weight=None, epochs=NUM_EPOCHS):

    data = ld.prepare_dataset("train")
    train_features, train_descriptions = data[0]
    test_features, test_descriptions = data[1]
    # prepare tokenizer
    tokenizer = gen.create_tokenizer(train_descriptions)
    dump(tokenizer, open('models/tokenizer.pkl', 'wb'))
    # index_word dict
    index_word = {v: k for k, v in tokenizer.word_index.items()}
    dump(index_word, open('models/index_word.pkl', 'wb'))

    VOCAB_SIZE = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % VOCAB_SIZE)

    MAX_CAP_LEN = MODEL.max_cap_len(train_descriptions)
    print('Description Length: %d' % MAX_CAP_LEN)

    embedding_matrix = Glove_word2vectorizer()
    model = MODEL.cnn_lstm_model_creater(img_pixel_len=IMG_PIXEL, img_data_size=IMG_SAMPLE_SIZE, img_embedding_dim=IMG_EMBEDDING_DIM,
                                         cap_pixel_len=MAX_CAP_LEN, cap_vocab_size=VOCAB_SIZE, cap_embedding_dim=CAP_EMBEDDING_DIM,)

    if weight != None:
        model.load_weights(weight)

    # define checkpoint callback
    filepath = 'models/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                  save_best_only=True, mode='min')

    steps = len(train_descriptions)
    val_steps = len(test_descriptions)

    train_generator = MODEL.data_generator(train_descriptions, train_features, tokenizer, max_length)
    val_generator = MODEL.data_generator(test_descriptions, test_features, tokenizer, max_length)

    model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=steps, verbose=1,
          callbacks=[checkpoint], validation_data=val_generator, validation_steps=val_steps)

    try:
        model.save('models/wholeModel.h5', overwrite=True)
        model.save_weights('models/weights.h5',overwrite=True)
    except:
        print("Error in saving model.")
    print("Training complete...\n")
