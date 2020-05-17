import pandas as pd
import numpy as np
import random
import configparser

random.seed(123)

config = configparser.ConfigParser()
config.read('../config.ini')

data_dir = config['PATHS']['data']
export_dir = data_dir + 'extracted_data/'

test_ratio = 0.1
dev_ratio = 0.2

if __name__ == "__main__":

    objects = pd.read_csv(export_dir+"all_objects.csv", index_col=0)
    object_ids = set(np.unique(objects.object_id.values))

    test_size = round(len(object_ids) * test_ratio)
    dev_size = round(len(object_ids) * dev_ratio)

    # sample objects for test set
    test_ids = set(random.sample(object_ids, test_size))
    available_objects = object_ids.difference(test_ids)
    # sample objects for train set
    dev_ids = set(random.sample(available_objects, dev_size))
    available_objects = available_objects.difference(dev_ids)
    # set remaining objects as train set
    train_ids = available_objects

    train_df = objects.loc[objects.object_id.isin(train_ids)]
    dev_df = objects.loc[objects.object_id.isin(dev_ids)]
    test_df = objects.loc[objects.object_id.isin(test_ids)]

    print('train items:', len(train_df))
    print('dev items:', len(dev_df))
    print('test items:', len(test_df))

    # Train/Test-Splits als CSV exportieren
    export = train_df.to_csv(export_dir+"train_df.csv")
    export = test_df.to_csv(export_dir+"test_df.csv")
    export = dev_df.to_csv(export_dir+"dev_df.csv")

    print('saved files to', export_dir)
