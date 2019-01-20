from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import numpy
import os
import argparse
import torch
import torch.nn as nn
import torch.utils.data as Data
from nltk.tokenize import WordPunctTokenizer
import model_cnn
import train


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=300, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()



#use train dataset as train and test set, just a little try
#load the dataset
data = open('amazon'+ os.path.sep +'train.lab').read()
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    labels.append(line)
    if i>=100000:
        break
data = open('amazon'+os.path.sep+'train.tgt').read()
for i, line in enumerate(data.split("\n")):
    content = line.split()
    texts.append(content)
    if i>=100000:
        break

# create a dataframe using texts and lables
# trainDF = pandas.DataFrame()
# trainDF['text'] = texts
# trainDF['label'] = labels

# split the dataset into training and validation datasets
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(texts, labels)

# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# load the pre-trained word-embedding vectors
embeddings_index = {}
# for i, line in enumerate(open('wiki-news-300d-1M.vec','r', encoding='UTF-8')):
#     values = line.split()
#     embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

import pickle
def save_obj(obj, name ):
    with open('obj'+ os.path.sep + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj' + os.path.sep + name + '.pkl', 'rb') as f:
        return pickle.load(f)
#save_obj(embeddings_index, 'dic')
embeddings_index = load_obj('dic')


word_index = {}
#words = WordPunctTokenizer().tokenize(data)
#words = WordPunctTokenizer().tokenize(open('amazon'+os.path.sep+'train.tgt').read())
words = []
for text in texts:
    for word in text:
        words.append(word)
print("safddasf")
words = list(set(words))
for i, values in enumerate(words):
    word_index[values] = i+1
print("word num{}".format(len(words)))
# from csdn
def text2seq(texts,maxlen=70):
    texts_with_id = numpy.zeros([len(texts), maxlen])
    for i in range(0, len(texts)):
        if i%100==0:
            print(i)
        if len(texts[i]) < maxlen:
            for j in range(0, len(texts[i])):
                if texts[i][j] in word_index:
                    texts_with_id[i][j] = word_index[texts[i][j]]
            for j in range(len(texts[i]), maxlen):
                texts_with_id[i][j] = 0
        else:
            for j in range(0, maxlen):
                if texts[i][j] in word_index:
                    texts_with_id[i][j] = word_index[texts[i][j]]
    return texts_with_id
train_seq_x = text2seq(train_x)
valid_seq_x = text2seq(valid_x)
train_seq_x = torch.from_numpy(numpy.asarray(train_seq_x, dtype = numpy.intp))
valid_seq_x = torch.from_numpy(numpy.asarray(valid_seq_x, dtype = numpy.intp))
train_y = torch.from_numpy(numpy.asarray(train_y))
valid_y = torch.from_numpy(numpy.asarray(valid_y))

# create token-embedding mapping
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print(embedding_matrix[10])

args.embed_num = len(word_index)
args.embedding_matrix = torch.from_numpy(embedding_matrix).float()
args.class_num = 5 


cnn = model_cnn.CNN(args)
if torch.cuda.is_available():
    torch.cuda.set_device(1)
    cnn = cnn.cuda()
    train_seq_x = train_seq_x.cuda()
    train_y = train_y.cuda()
    valid_seq_x = valid_seq_x.cuda()
    valid_y = valid_y.cuda()
    args.embedding_matrix = args.embedding_matrix.cuda()


train_dataset = Data.TensorDataset(train_seq_x, train_y)
train_loader = Data.DataLoader(
    dataset=train_dataset,      # torch TensorDataset format
    batch_size=args.batch_size,      # mini batch size
    shuffle=args.shuffle,             
    #num_workers=2,              
)


try:
    train.train(train_loader, valid_seq_x, valid_y, cnn, args)
except KeyboardInterrupt:
    print('\n' + '-' * 89)
    print('Exiting from training early')
