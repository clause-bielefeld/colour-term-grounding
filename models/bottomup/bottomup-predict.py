import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import configparser
import numpy as np
import os

config = configparser.ConfigParser()
config.read('../../config.ini')

data_dir = config['PATHS']['data']
array_input_dir = data_dir+'feature_arrays/'
model_input_dir = data_dir+'models/'
output_dir = data_dir+'prediction_arrays/'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
    print ('created path {path}'.format(path=output_dir))

batch_size = 128
num_classes = 11
epochs = 25

for colorspace in ['bgr']:
    print ('colorspace:', colorspace)

    model_file = 'bottomup_'+colorspace+'.h5'
    model = load_model(model_input_dir+model_file)
    print ('using model', model_file)

    input_file = 'histograms_'+colorspace+'_resampled.npz'
    import_arrays = np.load(array_input_dir+input_file)
    print ('importing arrays from', input_file)

    # exclude 1st column from every array (contains id)
    print ('\n\nAll Objects')
    test_x = import_arrays['test_x']
    test_y = import_arrays['test_y']
    dev_x = import_arrays['dev_x']
    dev_y = import_arrays['dev_y']
    add_x = import_arrays['add_x']
    add_y = import_arrays['add_y']
    x_test = test_x[:,1:]
    y_test = test_y[:,1:]
    x_dev = dev_x[:,1:]
    y_dev = dev_y[:,1:]
    x_add = add_x[:,1:]
    y_add = add_y[:,1:]
    print(x_test.shape[0], 'test samples')
    print(x_dev.shape[0], 'dev samples')
    print(x_add.shape[0], 'add samples')

    predict_dev_y = model.predict(x_dev)
    predict_test_y = model.predict(x_test)
    predict_add_y = model.predict(x_add)

    score = model.evaluate(x_dev, y_dev, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    results_file = 'results_bottomup_'+colorspace+'.npz'
    print ('writing to file', results_file)
    np.savez_compressed(output_dir+results_file,
        dev_y=dev_y,
        test_y = test_y,
        add_y = add_y,
        predict_dev_y = predict_dev_y,
        predict_test_y = predict_test_y,
        predict_add_y = predict_add_y
        )
