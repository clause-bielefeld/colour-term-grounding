import numpy as np
np.random.seed(123)

from numpy import zeros, asarray
import keras
from keras.layers import Input, Dense, Dropout,Embedding,Flatten
from keras.models import Model
from keras.utils.vis_utils import model_to_dot
from keras.optimizers import RMSprop
from keras.utils import plot_model
import configparser
import os
import pandas as pd
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read('../../config.ini')

data_dir = config['PATHS']['data']
input_dir = data_dir+'feature_arrays/'
output_dir = data_dir + 'models/'
batch_size = 128
num_classes = 11
epochs = 25

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
    print ('Pfad {path} angelegt'.format(path=output_dir))

def embeddings_reader(file_name):
    for line in open(file_name, "r"):
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        # only store relevant items in embeddings_index
        #if word in np.unique(x_type[:,1]):
        yield [word, coefs]

def plot_history(history):
    """ Plot training & validation accuracy values """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

# Gezippte Numpy-Archive importieren
topdown_arrays = np.load(input_dir+'object_types_resampled.npz', allow_pickle=True)

train_x = topdown_arrays['train_x']
train_y = topdown_arrays['train_y']
train_ids = topdown_arrays['train_y'][:,0]

dev_x = topdown_arrays['dev_x']
dev_y = topdown_arrays['dev_y']
dev_ids = topdown_arrays['dev_y'][:,0]

test_x = topdown_arrays['test_x']
test_y = topdown_arrays['test_y']
test_ids = topdown_arrays['test_y'][:,0]

add_x = topdown_arrays['add_x']
add_y = topdown_arrays['add_y']
add_ids = topdown_arrays['add_y'][:,0]
# # vortrainierte Word-Embeddings
# ------

# get types (as int)
x_type = np.concatenate([train_x[:,1:],dev_x[:,1:],test_x[:,1:],add_x[:,1:]])
x_type = np.argmax(x_type, axis=1)

obj_ids = np.concatenate([train_ids, dev_ids, test_ids, add_ids])
# match type-ints with object names
obj_df = pd.read_csv(data_dir+"extracted_data/all_objects.csv", index_col=0)
x_type = np.stack((x_type, np.array(obj_df.loc[obj_ids]['object_name'])), axis=1)

# load embeddings into memory
glove_path = data_dir+'embeddings/glove.6B/glove.6B.100d.txt'
embeddings_index = dict()
for word,coefs in embeddings_reader(glove_path):
    embeddings_index[word] = coefs
print('Loaded %s word vectors.' % len(embeddings_index))

# set vocab_size, get number of unique items. vocab_size: highest index +1
vocab_size = max(np.unique(x_type[:,0]))+1
print('size of vocabulary:', vocab_size)
print('unique objects:',len(np.unique(x_type[:,0])))

# load embeddings into memory
glove_path = data_dir+'embeddings/glove.6B/glove.6B.100d.txt'
embeddings_index = dict()
for word,coefs in embeddings_reader(glove_path):
    embeddings_index[word] = coefs
print('Loaded {s} word vectors.'.format(s=len(embeddings_index)))

# set vocab_size, get number of unique items. vocab_size: highest index +1
vocab_size = max(np.unique(x_type[:,0]))+1
print('size of vocabulary:', vocab_size)
print('unique objects:',len(np.unique(x_type[:,0])))

# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for i,word in x_type:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# words not included in GloVe
missing_words = np.unique([w for _,w in x_type if w not in embeddings_index.keys()])
print ('words not included in GloVe:',missing_words)

##########
# Model
##########

# Inputs
inputs = Input(shape=(1,), name='input')

embedding = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=1, trainable=False, name='embedding')(inputs)
flatten = Flatten(name='flatten')(embedding)
post_embedding = Dense(24, activation='relu')(flatten)
dropout = Dropout(0.2)(post_embedding)

predictions = Dense(num_classes, activation='softmax', name='predictions')(dropout)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

#plot_model(model, to_file=data_dir+'images/topdown_model_plot.png', show_shapes=True, show_layer_names=False)

model.summary()

history = model.fit(train_x[:,1:].argmax(axis=1), train_y[:,1:],
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(dev_x[:,1:].argmax(axis=1), dev_y[:,1:]))
score = model.evaluate(dev_x[:,1:].argmax(axis=1), dev_y[:,1:], verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_file = 'topdown.h5'
print ('saving model to', model_file)
model.save(output_dir+model_file)

plot_history(history)
