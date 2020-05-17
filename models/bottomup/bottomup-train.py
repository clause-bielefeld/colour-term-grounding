import numpy as np
np.random.seed(123)

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout,Embedding,concatenate,Flatten
from keras.utils.vis_utils import model_to_dot
from keras.optimizers import RMSprop
from keras.utils import plot_model
import configparser
import os

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
    print ('created path {path}'.format(path=output_dir))

for colorspace in ['bgr']:
    print ('colorspace:', colorspace)

    # the data, split between train and test sets
    input_file = 'histograms_'+colorspace+'_resampled.npz'
    import_arrays = np.load(input_dir+input_file, allow_pickle=True)
    print ('input dir:', input_file)

    # exclude 1st column from every array (contains id)
    x_train = import_arrays['train_x'][:,1:]
    y_train = import_arrays['train_y'][:,1:]
    x_test = import_arrays['dev_x'][:,1:]
    y_test = import_arrays['dev_y'][:,1:]

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Inputs
    inputs = Input(shape=(512,), name='input')

    x = Dense(240, activation='relu',name='dense_0')(inputs)
    x = Dropout(0.2, name='dropout_0')(x)
    x = Dense(24, activation='relu',name='dense_1')(x)
    x = Dropout(0.2, name='dropout_1')(x)

    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.summary()
    #plot_model(model, to_file=data_dir+'bottomup_model_plot.png', show_shapes=True, show_layer_names=False)

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(learning_rate=0.001),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model_file = 'bottomup_'+colorspace+'.h5'
    print ('saving model to', model_file)
    model.save(output_dir+model_file)
