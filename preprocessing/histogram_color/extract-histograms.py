import pandas as pd
import numpy as np
import os
import sys
import configparser

sys.path.append(os.path.abspath('../../utils'))
import preprocessing

config = configparser.ConfigParser()
config.read('../../config.ini')

vg_json = config['PATHS']['vg-json']
data_dir = config['PATHS']['data']
image_dir = config['PATHS']['vg-images']

input_dir = data_dir + 'extracted_data/'
output_dir = data_dir + 'feature_arrays/'

basic_colors = preprocessing.basic_colors()

if __name__ == "__main__":

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        print('created path {path}'.format(path=output_dir))

    train_df = pd.read_csv(input_dir+"train_df.csv", index_col=0)
    test_df = pd.read_csv(input_dir+"test_df.csv", index_col=0)
    dev_df = pd.read_csv(input_dir+"dev_df.csv", index_col=0)

    print('one hot encoding for color names')
    train_df = pd.get_dummies(train_df, prefix=['color'], columns=['color'])
    test_df = pd.get_dummies(test_df, prefix=['color'], columns=['color'])
    dev_df = pd.get_dummies(dev_df, prefix=['color'], columns=['color'])

    col_list = train_df.columns.tolist()

    # reset index (for printing)
    train_df = train_df.reset_index(drop=False)
    test_df = test_df.reset_index(drop=False)
    dev_df = dev_df.reset_index(drop=False)

    for colorspace in ['bgr']:  # other color spaces: 'hsv','lab'

        print('get histograms and build feature arrays for train split')
        train_x, train_y = preprocessing.x_y_histograms(train_df, image_dir, convert=colorspace)
        print('get histograms and build feature arrays for test split')
        test_x, test_y = preprocessing.x_y_histograms(test_df, image_dir, convert=colorspace)
        print('get histograms and build feature arrays for dev split')
        dev_x, dev_y = preprocessing.x_y_histograms(dev_df, image_dir, convert=colorspace)

        # save results
        export_filename = 'histograms_'+colorspace+'.npz'
        np.savez_compressed(
            output_dir+export_filename,
            train_x=train_x, train_y=train_y,
            test_x=test_x, test_y=test_y,
            dev_x=dev_x, dev_y=dev_y
        )
    for colorspace in ['bgr']:  # 'hsv','lab'
        import_arrays = np.load(output_dir+'histograms_'+colorspace+'.npz')
        print('results '+colorspace+':')
        print('shape train_x:', import_arrays['train_x'].shape)
        print('shape train_y:', import_arrays['train_y'].shape)
        print('shape test_x:', import_arrays['test_x'].shape)
        print('shape test_y:', import_arrays['test_y'].shape)
        print('shape dev_x:', import_arrays['dev_x'].shape)
        print('shape dev_y:', import_arrays['dev_y'].shape)
