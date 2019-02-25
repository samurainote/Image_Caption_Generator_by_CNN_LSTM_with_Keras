

from pickle import load
import argparse

def load_doc(filename):
    file = open(filename, "r")
    text = file.read()
    file.close()
    return text

def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split("\n"):
        if len(line) < 1:
            continue
        identifier = line.split(".")[0]
        dataset.append(identifier)
    return set(dataset)


def train_test_spliter(dataset):
    ordered = sorted(dataset)
    train_size = round(len(ordered) * 0.7)
    test_size = len(ordered) - train_size
    train_data = set(ordered[:train_size])test_size
    test_data = set(ordered[train_size:test_size])
    return train_data, test_data


def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        # データセットの中に画像idが存在する
        if image_id in dataset:
            # 画像idが今作りたい辞書に存在しない
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = "BOS " + " ".join(image_desc) + " EOS"
            descriptions[image_id].append(desc)
    return descriptions


def load_image_feature(filename, dataset):
    all_features = load(open(filename, "rb"))
    # filter features
    features = {key: all_features[key] for key in dataset}
    return features

def prepare_dataset(data='dev'):
    assert data in ['dev', 'train', 'test']

    train_features = None
    train_descriptions = None

    if data == 'dev':
    # load dev set (1K)
        filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
        dataset = load_set(filename)
        print('Dataset: %d' % len(dataset))

        # train-test split
        train, test = train_test_split(dataset)
        #print('Train=%d, Test=%d' % (len(train), len(test)))

        # descriptions
        train_descriptions = load_clean_descriptions('models/descriptions.txt', train)
        test_descriptions = load_clean_descriptions('models/descriptions.txt', test)
        print('Descriptions: train=%d, test=%d' % (len(train_descriptions), len(test_descriptions)))

        # photo features
        train_features = load_photo_features('models/features.pkl', train)
        test_features = load_photo_features('models/features.pkl', test)
        print('Photos: train=%d, test=%d' % (len(train_features), len(test_features)))

    elif data == 'train':
    # load training dataset (6K)
        filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
        train = load_set(filename)

        filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
        test = load_set(filename)
        print('Dataset: %d' % len(train))

        # descriptions
        train_descriptions = load_clean_descriptions('models/descriptions.txt', train)
        test_descriptions = load_clean_descriptions('models/descriptions.txt', test)
        print('Descriptions: train=%d, test=%d' % (len(train_descriptions), len(test_descriptions)))

        # photo features
        train_features = load_photo_features('models/features.pkl', train)
        test_features = load_photo_features('models/features.pkl', test)
        print('Photos: train=%d, test=%d' % (len(train_features), len(test_features)))

    elif data == 'test':
        # load test set
        filename = 'Flickr8k_text/Flickr_8k.testImages.txt'
        test = load_set(filename)
        print('Dataset: %d' % len(test))
        # descriptions
        test_descriptions = load_clean_descriptions('models/descriptions.txt', test)
        print('Descriptions: test=%d' % len(test_descriptions))
        # photo features
        test_features = load_photo_features('models/features.pkl', test)
        print('Photos: test=%d' % len(test_features))

      return (train_features, train_descriptions), (test_features, test_descriptions)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate dataset features')
  parser.add_argument("-t", "--train", action='store_const', const='train',
    default = 'dev', help="Use large 6K training set")
  args = parser.parse_args()
  prepare_dataset(args.train)
