
"""
Image Data: Flickr8k_Dataset.zip (1 Gigabyte) An archive of all photographs.
Description Data: Flickr8k_text.zip (2.2 Megabytes) An archive of all text descriptions for photographs.

How data looks like:
>>> Image Data
990890291_afc72be141.jpg
99171998_7cc800ceef.jpg
99679241_adc853a5c0.jpg
997338199_7343367d7f.jpg
997722733_0cb5439472.jpg

>>> Description Data
1305564994_00513f9a5b.jpg#0 A man in street racer armor be examine the tire of another racer 's motorbike .
1305564994_00513f9a5b.jpg#1 Two racer drive a white bike down a road .
1305564994_00513f9a5b.jpg#2 Two motorist be ride along on their vehicle that be oddly design and color .
1305564994_00513f9a5b.jpg#3 Two person be in a small race car drive by a green hill .
1305564994_00513f9a5b.jpg#4 Two person in race uniform in a street car .
"""

from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Reshape, Concatenate
import numpy as np
import string
from progressbar import progressbar
from keras.models import Model

# ========================================================================
# Get image features from pretrained VGG16
# ========================================================================

# load an image from filepath
def load_image_datas(path):
    img = load_img(path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) # GBR pixel,pixel,color
    img = preprocess_input(img)
    return np.asarray(img)

# 事前に学習ずみの画像内のオブジェクトを認識するための特徴をVGG16より抽出する
# 例えば、Flicker8内の新しい画像Aに対してそ子に映る猫を猫だと認識してキャプションに組み込む
def extract_image_features(directory):
    model = VGG16()
    model.layers.pop() # モデルを初期化する
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    print(model.summary())
    features = dict()
    for name in progressbar(listdir(directory)):
        # ignore README
        if name == 'README.md':
          continue
        file_path = directory + '/' + name
        image = load_image_datas(file_path)
        # extract features
        feature = model.predict(image, verbose=0)
        # get image id
        image_id = name.split('.')[0] # 997722733_0cb5439472.jpg
        image_id_features[image_id] = feature # dictionary
    return image_id_features


# ========================================================================
# Create vocablary and padding sentence from image description data
# ========================================================================

def load_desc_file(filename):
    file = open(filename, "r")
    text = file.read()
    file.close()
    return text

# { image_id: desc1, desc2, desc3, desc4, desc5 }　という辞書を作る
def load_descriptions(descriptions):
    imgid_desc_dict = dict()
    for line in descriptions.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:] # 1305564994_00513f9a5b.jpg#0 A man in street racer armor be examine the tire of another racer 's motorbike .
        image_id = image_id.split(".")[0]
        image_desc = " ".join(image_desc)
        if image_id not in imgid_desc_dict:
            imgid_desc_dict[image_id].append(image_desc)
    return imgid_desc_dict

# descriptions = load_descriptions()
def clean_descriptions(descriptions):
    punct = str.maketrans("", "", string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc] # 小文字化
            desc = [word.translate(punct) for word in desc] # 句読点の削除
            desc = [word for word in desc if len(word)>1] # 's' and 'a'　とかは削除
            desc = [word for word in desc if word.isalpha()] # 数字は削除
            desc_list[i] =  " ".join(desc) # 改めて文に戻す

# convert the loaded descriptions into a vocabulary of words
def create_vocabulary(descriptions):
    all_desc = set() # 重複を持たない辞書、順序がない！
    for key in descriptions.keys():
        for desc in descriptions[key]:
            all_desc.update(desc.split())
    return all_desc

 # save descriptions to file, one per line
def save_descriptions(descriptions, filename):
  lines = list()
  for key, desc_list in descriptions.items():
    for desc in desc_list:
      lines.append(key + ' ' + desc)
  data = '\n'.join(lines)
  file = open(filename, 'w')
  file.write(data)
  file.close()


# ========================================================================
# extract features from all images
# ========================================================================

directory = "Flickr8k_Dataset"
features = extract_image_features(directory)
print("Extracted Features: %d" % len(features))
# save to file
dump(features, open("models/features.pkl", "wb"))

# prepare descriptions
filename = "Flickr8k_text/Flickr8k.token.txt"
# load descriptions
text = load_text_file(filename)
# parse descriptions
descriptions = load_descriptions(text)
print("Loaded: %d" % len(descriptions))
# clean descriptions
clean_descriptions(descriptions)
# summarize vocabulary
vocabulary = create_vocabulary(descriptions)
print("Vocabulary Size: %d" % len(vocabulary))
# save to file
save_descriptions(descriptions, "models/descriptions.txt")
