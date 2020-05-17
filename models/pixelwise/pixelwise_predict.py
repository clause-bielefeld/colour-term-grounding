import pandas as pd
import numpy as np
import sys
import os
import configparser
import skimage
from skimage import io

sys.path.append(os.path.abspath('../../utils'))
import preprocessing

config = configparser.ConfigParser()
config.read('../../config.ini')
vg_json = config['PATHS']['vg-json']
data_dir = config['PATHS']['data']
image_dir = config['PATHS']['vg-images']
model_dir = data_dir + 'models/'
output_dir = data_dir+'prediction_arrays/'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
    print('created dir {path}'.format(path=output_dir))

source_dir = data_dir+'extracted_data/'
w2c_datasets = np.load(model_dir+'pixelwise_lookup_tables.npz')


def color_pixelwise(img, w2c, bb=False):
    # cf. file im2c.m from http://lear.inrialpes.fr/people/vandeweijer/code/ColorNaming.tar
    # Input: image path and w2c matrix (RGB values -> colour names)

    img = io.imread(img)

    # grayscale images
    if len(img.shape) < 3:
        img = skimage.color.gray2rgb(img, alpha=None)

    if bb:
        if type(bb) == list:
            bb = {
                'h': bb[0],
                'w': bb[1],
                'x': bb[2],
                'y': bb[3]
            }
        img = img[bb['y']:bb['y']+bb['h'], bb['x']:bb['x']+bb['w']]

    # split rgb channels
    RR = img[:, :, 0]
    GG = img[:, :, 1]
    BB = img[:, :, 2]

    index_img = np.array(
        # R values (32 Bins)
        np.floor(RR/8)+
        # G values (32 Bins)
        32* np.floor(GG/8)+
        # B values (32 Bins)
        32*32*np.floor(BB/8)
    )
    # initialize array for probability distribution over colour terms
    clr_distribution = np.zeros(11)

    # add probability distribution for every index to clr_distribution
    for pxl_index in index_img.ravel():
        clr_distribution = clr_distribution + w2c[int(pxl_index)][3:]

    # normalize clr_distribution
    clr_distribution = clr_distribution / len(index_img.ravel())

    return clr_distribution


def rows_pixelwise_classification(row,  w2c):
    bb = {
        'h': row.bb_h,
        'w': row.bb_w,
        'x': row.bb_x,
        'y': row.bb_y
    }
    img_path = image_dir + str(row.image_id) + '.jpg'
    return (np.append(row.Index, color_pixelwise(img_path, w2c, bb)))


def dataframe_pixelwise_classification(df, w2c):
    results = np.empty((0, 12))
    for row in df.itertuples():
        res = rows_pixelwise_classification(row, w2c)
        res = np.reshape(res, (1, 12))
        results = np.append(results, res, axis=0)
        if results.shape[0] % 1000 == 0:
            print(results.shape[0], '/', len(df))
    return results


if __name__ == "__main__":
    test_df = pd.read_csv(source_dir+"test_df.csv", index_col=0)
    dev_df = pd.read_csv(source_dir+"dev_df.csv", index_col=0)

    print('Test set shape:', test_df.shape)
    print('Dev set shape:', dev_df.shape)

    print('Starting Classification')
    dev_w2c = dataframe_pixelwise_classification(dev_df, w2c_datasets['w2c'])
    test_w2c = dataframe_pixelwise_classification(test_df, w2c_datasets['w2c'])
    # dev_chip_w2c = dataframe_pixelwise_classification(dev_df, w2c_datasets['chip_w2c'])
    # test_chip_w2c = dataframe_pixelwise_classification(test_df, w2c_datasets['chip_w2c'])

    # Als Datei exportieren
    print('write numpy arrays to file')

    export_filename = 'results_pixelwise.npz'
    np.savez_compressed(
        output_dir+export_filename,
        test_w2c=test_w2c,
        dev_w2c=dev_w2c,
        # test_chip_w2c = test_chip_w2c,
        # dev_chip_w2c = dev_chip_w2c
    )

    print('shapes:')
    print('test_w2c:', test_w2c.shape)
    print('dev_w2c:', dev_w2c.shape)
    # print('test_chip_w2c', test_chip_w2c.shape)
    # print('dev_chip_w2c', dev_chip_w2c.shape)
