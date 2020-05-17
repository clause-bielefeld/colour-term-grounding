import os
import sys
import pandas as pd
import numpy as np
import requests
import itertools
import math
import configparser

config = configparser.ConfigParser()
config.read('../../config.ini')

sys.path.append(os.path.abspath('../../utils'))
import preprocessing

vg_json = config['PATHS']['vg-json']
vg_json_export = config['PATHS']['data']
image_dir = config['PATHS']['vg-images']


def gauss(x, mu=0.0, sigma=1.0):
    """
    Return the value of the Gaussian probability function with mean mu
    and standard deviation sigma at the given x value.
    """
    # aus https://introcs.cs.princeton.edu/python/22module/gaussian.py.html
    x = float(x - mu) / sigma
    return math.exp(-x*x/2.0) / math.sqrt(2.0*math.pi) / sigma


###########################
# chip based data set #
###########################

# retrieve text file
r = requests.get('http://www.cvc.uab.es/color_naming/MembershipValues_sRGB.txt')
data = r.text
data = data.split('\r\n')
# last line is empty
data = data[:-1]

# cf. http://www.cvc.uab.es/color_naming/
columns = 'r g b red orange brown yellow green blue purple pink white gray black'.split()
columns_reordered = 'r g b'.split() + preprocessing.basic_colors()
df = pd.DataFrame(columns=columns)

for line in data:
    s = pd.Series(line.split('\t'), index = columns)
    df = df.append(s,ignore_index=True)

df = df.astype('float64')

# initialize list with rgb bins
l = []
n = 3.5
for i in range(32):
    l.append(n)
    n += 8
rgb_bins = list(itertools.product(l, l, l))
rgb_bins = sorted(rgb_bins, key=lambda x: (x[2], x[1], x[0]))

chip_w2c = np.empty((0, 14))
# iterate over rgb bins
for rgb_i in rgb_bins:

    # initialize np-array for colour name membership values
    p_i = np.zeros(11)

    for j in df.to_numpy():
        # split rgb values from membership values
        rgb_j = j[:3]
        p_j = j[3:]
        # Euclidean distance between the rgb values of i and j
        distance = np.linalg.norm(rgb_i-rgb_j)
        p_j = p_j * gauss(distance, sigma=5)
        #p_j = p_j / len(df.to_numpy())
        p_i += p_j
    p_i = p_i / p_i.sum()
    i = np.append(rgb_i, p_i)
    chip_w2c = np.append(chip_w2c, [i], axis=0)

chip_w2c_df = pd.DataFrame(chip_w2c, columns=columns)
chip_w2c_df = chip_w2c_df[columns_reordered]

# als np-Array exportieren
chip_w2c_arr = chip_w2c_df.to_numpy()

##########################################################
# data from J. van de Weijer, C. Schmid, J. Verbeek 2007 #
##########################################################

data_dir = vg_json_export + 'pixelwise/'

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if 'ColorNaming' in os.listdir(data_dir):
    data_dir += 'ColorNaming/'
else:
    # extract tar file
    import requests, tarfile, io
    url = 'http://lear.inrialpes.fr/people/vandeweijer/code/ColorNaming.tar'
    r = requests.get(url)
    b = io.BytesIO(r.content)
    tar = tarfile.TarFile(fileobj=b)
    tar.extractall(path=data_dir)
    data_dir += 'ColorNaming/'

w2c = []
# read w2c file
f = open(data_dir+'w2c.txt', 'r')
for line in f:
    line = line.replace(' \n', '')
    w2c.append(line.split())

# np array from w2c list
w2c_arr = np.array(w2c).astype('float64')

##########
# export #
##########

np.savez_compressed(vg_json_export+'pixelwise_data.npz',
    w2c=w2c_arr,
    chip_w2c=chip_w2c_arr
    )

loaded = np.load(vg_json_export+'pixelwise_data.npz')

print('w2c-Array:', loaded['w2c'].shape)
print('chip-based w2c-Array:', loaded['chip_w2c'].shape)
