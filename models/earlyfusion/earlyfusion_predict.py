from keras.models import load_model
import configparser
import numpy as np
import os
import pandas as pd

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

# Gezippte Numpy-Archive importieren
bottomup_arrays = np.load(array_input_dir+'histograms_bgr_resampled.npz', allow_pickle=True)
topdown_arrays = np.load(array_input_dir+'object_types_resampled.npz', allow_pickle=True)

test_bu_x = bottomup_arrays['test_x']
test_td_x = topdown_arrays['test_x']
test_y = topdown_arrays['test_y']
test_ids = topdown_arrays['test_x'][:,0]

test_bu_x = pd.DataFrame(test_bu_x, index=test_bu_x[:,0]).loc[test_ids].to_numpy()

dev_bu_x = bottomup_arrays['dev_x']
dev_td_x = topdown_arrays['dev_x']
dev_y = topdown_arrays['dev_y']
dev_ids = topdown_arrays['dev_y'][:,0]

dev_bu_x = pd.DataFrame(dev_bu_x, index=dev_bu_x[:,0]).loc[dev_ids].to_numpy()

add_bu_x = bottomup_arrays['add_x']
add_td_x = topdown_arrays['add_x']
add_y = topdown_arrays['add_y']
add_ids = topdown_arrays['add_x'][:,0]

add_bu_x = pd.DataFrame(add_bu_x, index=add_bu_x[:,0]).loc[add_ids].to_numpy()

print(test_bu_x.shape[0], 'test samples')
print(dev_bu_x.shape[0], 'dev samples')
print(add_bu_x.shape[0], 'add samples')
print ('Test-IDs identisch:', np.array_equal(test_bu_x[:,0], test_td_x[:,0]))
print ('Dev-IDs identisch:', np.array_equal(dev_bu_x[:,0], dev_td_x[:,0]))
print ('Add-IDs identisch:', np.array_equal(add_bu_x[:,0], add_td_x[:,0]))

###
# Model Files
###

model_file = 'earlyfusion_model.h5'
model = load_model(model_input_dir+model_file)

predict_dev_y = model.predict([dev_bu_x[:,1:], dev_td_x[:,1:].argmax(axis=1)])
predict_test_y = model.predict([test_bu_x[:,1:], test_td_x[:,1:].argmax(axis=1)])
predict_add_y = model.predict([add_bu_x[:,1:], add_td_x[:,1:].argmax(axis=1)])

score = model.evaluate([dev_bu_x[:,1:], dev_td_x[:,1:].argmax(axis=1)], dev_y[:,1:], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

###
# Output Files
###

results_file = 'results_earlyfusion.npz'
print ('writing to file', results_file)
np.savez_compressed(output_dir+results_file,
    dev_y=dev_y,
    test_y = test_y,
    add_y = add_y,
    predict_dev_y = predict_dev_y,
    predict_test_y = predict_test_y,
    predict_add_y = predict_add_y
    )
