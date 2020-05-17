from keras.models import load_model
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
    print ('Pfad {path} angelegt'.format(path=output_dir))

batch_size = 128
num_classes = 11
epochs = 25

# import zipped numpy archives
topdown_arrays = np.load(array_input_dir+'object_types_resampled.npz', allow_pickle=True)

test_x = topdown_arrays['test_x']
test_y = topdown_arrays['test_y']
test_ids = topdown_arrays['test_x'][:,0]

dev_x = topdown_arrays['dev_x']
dev_y = topdown_arrays['dev_y']
dev_ids = topdown_arrays['dev_y'][:,0]

add_x = topdown_arrays['add_x']
add_y = topdown_arrays['add_y']
add_ids = topdown_arrays['add_x'][:,0]

print(test_x.shape[0], 'test samples')
print(dev_x.shape[0], 'dev samples')
print(add_x.shape[0], 'add samples')

m_name = 'topdown'
model_file = m_name + '.h5'
model = load_model(model_input_dir+model_file)

predict_dev_y = model.predict(dev_x[:,1:].argmax(axis=1))
predict_test_y = model.predict(test_x[:,1:].argmax(axis=1))
predict_add_y = model.predict(add_x[:,1:].argmax(axis=1))

score = model.evaluate(dev_x[:,1:].argmax(axis=1), dev_y[:,1:], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

results_file = 'results_' + m_name + '.npz'
print ('writing to file', results_file)
np.savez_compressed(output_dir+results_file,
    dev_y=dev_y,
    test_y = test_y,
    add_y = add_y,
    predict_dev_y = predict_dev_y,
    predict_test_y = predict_test_y,
    predict_add_y = predict_add_y
    )
