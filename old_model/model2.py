
from vgg16 import VGG16
from keras.applications import inception_v3
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
from keras.preprocessing import image, sequence
from keras.callbacks import ModelCheckpoint
import pickle

MAX_CAP_LEN = None
VOCAB_SIZE = None
IDX2WORD = None
self.WORD2IDX = None
self.NUM_SAMPLES = None
self.ENCODED_IMAGES = pickle.load(open( "encoded_images.p", "rb" ))
IMAGE_EMBEDDING_DIM = 128

"""
TITLE: Surpervised Image Captioning
GOAL:
INPUT: Image Data
OUTPUT: Sentence which explain given image

ARCHITECTURE: CNN-LSTM Model
1. LSTM

"""
