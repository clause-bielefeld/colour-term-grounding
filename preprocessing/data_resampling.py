import pandas as pd
import numpy as np
import sys
import os
import configparser
import collections
from imblearn.under_sampling import RandomUnderSampler
from nltk.corpus import wordnet as wn

sys.path.append(os.path.abspath('../utils'))
import preprocessing

config = configparser.ConfigParser()
config.read('../config.ini')
vg_json = config['PATHS']['vg-json']
data_dir = config['PATHS']['data']
image_dir = config['PATHS']['vg-images']

input_dir = data_dir + 'extracted_data/'
output_dir = data_dir + 'feature_arrays/'

random_state = 123
min_num_add = 50

train_instances_per_color = 10000


def embeddings_reader(file_name):
    for line in open(file_name, "r"):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        yield [word, coefs]


def get_hyponyms(synset):
    """
    https://stackoverflow.com/questions/15330725/how-to-get-all-the-hyponyms-of-a-word-synset-in-python-nltk-and-wordnet
    """
    hyponyms = set()
    hyponyms.update({synset})
    for hyponym in synset.hyponyms():
        hyponyms |= set(get_hyponyms(hyponym))
    return hyponyms | set(synset.hyponyms())


if __name__ == "__main__":

    #############################################
    # 1) features for top-down processing:      #
    # - select frequent objects for train/test  #
    #   (more than 100 instances)               #
    # - select additional objects               #
    #   (more than 50, less than 100 instances, #
    #   not included in train/test)             #
    # - one-hot encode object types + colours   #
    #############################################

    # create output directory, if it doesn't exist
    if not os.path.isdir(output_dir):
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        os.mkdir(output_dir)
        print('created path {path}'.format(path=output_dir))

    # get extracted data
    objects = pd.read_csv(input_dir+"all_objects.csv", index_col=0)
    # sort out objects with empty or invalid (non-string) names
    objects = objects.loc[
        objects.apply(lambda x: type(x.object_name) == str and x.object_name != '', axis=1)
        ]

    # get objects with more than 50 instances
    frequent_objects, _, _, _ = preprocessing.freq_cdo_cno(objects, num_cdos=100, num_cnos=100, min_num=min_num_add)
    frequent_objects_df = objects['color object_name'.split()].loc[objects.object_name.isin(frequent_objects)]
    # get objects with more than 100 instances
    frequent_objects_traintest, _, _, _ = preprocessing.freq_cdo_cno(objects, num_cdos=100, num_cnos=100, min_num=100)
    additional_objects = [obj for obj in frequent_objects if obj not in frequent_objects_traintest]

    # get GloVe embeddings
    glove_path = data_dir+'embeddings/glove.6B/glove.6B.100d.txt'
    embeddings_index = dict()
    for word, coefs in embeddings_reader(glove_path):
        embeddings_index[word] = coefs
    # sort out additional objects not present in the GloVe embeddings
    additional_objects = [word for word in additional_objects if word in embeddings_index.keys()]

    # one-hot encode object types
    frequent_objects_df['_types'] = frequent_objects_df['object_name']
    frequent_objects_df = pd.get_dummies(frequent_objects_df, columns=['_types'], prefix='type')

    # get train, dev, test splits
    train_index = pd.read_csv(input_dir+"train_df.csv", index_col=0).index.intersection(frequent_objects_df.index)
    test_index = pd.read_csv(input_dir+"test_df.csv", index_col=0).index.intersection(frequent_objects_df.index)
    dev_index = pd.read_csv(input_dir+"dev_df.csv", index_col=0).index.intersection(frequent_objects_df.index)
    train_df = frequent_objects_df.loc[train_index]
    test_df = frequent_objects_df.loc[test_index]
    dev_df = frequent_objects_df.loc[dev_index]

    # create DataFrame for additional / unseen objects
    additional_ids = objects.loc[objects.object_name.isin(additional_objects)].index
    add_df = frequent_objects_df.loc[frequent_objects_df.object_name.isin(additional_objects)]

    # remove additional objects from train, test and dev sets
    train_df = train_df[~train_df.index.isin(additional_ids)]
    dev_df = dev_df[~dev_df.index.isin(additional_ids)]
    test_df = test_df[~test_df.index.isin(additional_ids)]

    # drop object_name column with nominal data
    [train_df, test_df, dev_df, add_df] = [df.drop(columns=['object_name']) for df in [train_df, test_df, dev_df, add_df]]

    # one-hot encode object colors
    train_y, test_y, dev_y, add_y = [df['color'] for df in [train_df, test_df, dev_df, add_df]]
    train_y, test_y, dev_y, add_y = [pd.get_dummies(df, prefix='color') for df in [train_y, test_y, dev_y, add_y]]
    for df in [train_y, test_y, dev_y, add_y]:
        df.insert(0, 'index', df.index, allow_duplicates=False)

    # delete color column with nominal data
    train_x, test_x, dev_x, add_x = [df.drop(columns=['color']) for df in [train_df, test_df, dev_df, add_df]]
    for df in [train_x, test_x, dev_x, add_x]:
        df.insert(0, 'index', df.index, allow_duplicates=False)

    # convert into numpy arrays
    train_x, train_y, test_x, test_y, dev_x, dev_y, add_x, add_y = [df.to_numpy() for df in [train_x, train_y, test_x, test_y, dev_x, dev_y, add_x, add_y]]

    # export indices
    export_filename = 'additional_ids.npz'
    np.savez_compressed(
        output_dir+export_filename,
        ids=np.asarray(additional_ids)
    )

    # export raw feature arrays for top down classifier
    export_filename = 'object_types.npz'
    np.savez_compressed(
        output_dir+export_filename,
        train_x=train_x,
        train_y=train_y,
        dev_x=dev_x,
        dev_y=dev_y,
        test_x=test_x,
        test_y=test_y,
        add_x=add_x,
        add_y=add_y
    )

    ####################################################################
    # 2) get rid of person hyponyms, Dev/Test random undersampling     #
    #    (apply undersampling to top-down features,                    #
    #     use the resulting ids for bottom-up features)                #
    ####################################################################

    input_dir = data_dir + 'feature_arrays/'
    num_classes = 11

    all_obj = pd.read_csv(data_dir+"extracted_data/all_objects.csv", index_col=0)
    train_obj = pd.read_csv(data_dir+"extracted_data/train_df.csv", index_col=0)
    test_obj = pd.read_csv(data_dir+"extracted_data/test_df.csv", index_col=0)
    dev_obj = pd.read_csv(data_dir+"extracted_data/dev_df.csv", index_col=0)

    # remove hyponyms of 'person.n.01'
    person_hyponyms = get_hyponyms(wn.synset('person.n.01'))
    person_words = []
    for synset in person_hyponyms:
        person_words += [l.name().replace('_', ' ') for l in synset.lemmas()]
    no_person_index = all_obj.loc[np.logical_not(all_obj.object_name.isin(person_words))].index.to_list()

    test_ids = []
    dev_ids = []
    add_ids = []

    # begin with object_types; since it only contains frequent objects
    for input_file in ['object_types', 'histograms_bgr']:

        import_arrays = np.load(input_dir+input_file+'.npz')

        # TRAIN-SET
        train_x = import_arrays['train_x']
        train_y = import_arrays['train_y']

        print('{file}: filtering train set. Original shapes:'.format(file=input_file), train_x.shape, train_y.shape)

        # remove person hyponyms
        train_x = pd.DataFrame(train_x).loc[pd.DataFrame(train_x)[0].isin(no_person_index)].to_numpy()
        train_y = pd.DataFrame(train_y).loc[pd.DataFrame(train_y)[0].isin(no_person_index)].to_numpy()

        print('{file}: train set shapes:'.format(file=input_file), train_x.shape, train_y.shape)

        # DEV-SET
        dev_x = import_arrays['dev_x']
        dev_y = import_arrays['dev_y']

        print('{file}: filtering and resampling dev set. Original shapes:'.format(file=input_file), dev_x.shape, dev_y.shape)

        if len(dev_ids) == 0:

            # remove person hyponyms
            dev_x = pd.DataFrame(dev_x).loc[pd.DataFrame(dev_x)[0].isin(no_person_index)].to_numpy()
            dev_y = pd.DataFrame(dev_y).loc[pd.DataFrame(dev_y)[0].isin(no_person_index)].to_numpy()

            print('random undersampling')
            rus = RandomUnderSampler(random_state=random_state)
            dev_x, dev_y = rus.fit_resample(dev_x, dev_y[:, 1:].argmax(axis=1))
            # dev_y from int to one-hot
            dev_y = np.eye(num_classes)[dev_y]
            # add ids to dev_y
            dev_y = np.append(dev_x[:, 0:1], dev_y, axis=1)

            dev_ids = dev_y[:, 0].ravel()

        else:
            print('select entries from dev_ids')
            dev_x = pd.DataFrame(dev_x, index=dev_x[:, 0]).loc[dev_ids].to_numpy()
            dev_y = pd.DataFrame(dev_y, index=dev_y[:, 0]).loc[dev_ids].to_numpy()

            instances_per_color = collections.Counter(dev_y[:, 1:].argmax(axis=1)).values()
            if min(instances_per_color) != max(instances_per_color):
                raise ValueError('{file}: colors not equally distributed in dev set'.format(file=input_file))

        print('{file}: dev set shapes:'.format(file=input_file), dev_x.shape, dev_y.shape)

        # TEST-SET
        test_x = import_arrays['test_x']
        test_y = import_arrays['test_y']

        print('{file}: filtering and resampling test set. Original shapes:'.format(file=input_file), test_x.shape, test_y.shape)

        if len(test_ids) == 0:

            # remove person hyponyms
            test_x = pd.DataFrame(test_x).loc[pd.DataFrame(test_x)[0].isin(no_person_index)].to_numpy()
            test_y = pd.DataFrame(test_y).loc[pd.DataFrame(test_y)[0].isin(no_person_index)].to_numpy()

            print('random undersampling')
            rus = RandomUnderSampler(random_state=random_state)
            test_x, test_y = rus.fit_resample(test_x, test_y[:, 1:].argmax(axis=1))
            # test_y von Integer zu One-Hot-Encoding
            test_y = np.eye(num_classes)[test_y]
            # IDs zu test_y hinzufügen
            test_y = np.append(test_x[:, 0:1], test_y, axis=1)

            test_ids = test_y[:, 0].ravel()

        else:
            print('select entries from test_ids')
            test_x = pd.DataFrame(test_x, index=test_x[:, 0]).loc[test_ids].to_numpy()
            test_y = pd.DataFrame(test_y, index=test_y[:, 0]).loc[test_ids].to_numpy()

            instances_per_color = collections.Counter(test_y[:, 1:].argmax(axis=1)).values()
            if min(instances_per_color) != max(instances_per_color):
                raise ValueError('{file}: colors not equally distributed in test set'.format(file=input_file))

        print('{file}: test set shapes:'.format(file=input_file), test_x.shape, test_y.shape)

        # ADD-SET

        try:
            add_x = import_arrays['add_x']
            add_y = import_arrays['add_y']
        except:
            if input_file == 'histograms_bgr':
                # additional objects not specified for visual feature arrays
                pass
            else:
                break

        print('{file}: filtering and resampling test set. Original shapes:'.format(file=input_file), add_x.shape, add_y.shape)

        if len(add_ids) == 0:
            # remove person hyponyms
            add_x = pd.DataFrame(add_x).loc[pd.DataFrame(add_x)[0].isin(no_person_index)].to_numpy()
            add_y = pd.DataFrame(add_y).loc[pd.DataFrame(add_y)[0].isin(no_person_index)].to_numpy()

            add_ids = add_y[:, 0].ravel()

        else:
            if input_file == 'histograms_bgr':
                # retrieve additional add items from train, test and dev sets
                add_x = np.concatenate([import_arrays['train_x'], import_arrays['dev_x'], import_arrays['test_x']])
                add_y = np.concatenate([import_arrays['train_y'], import_arrays['dev_y'], import_arrays['test_y']])

            # select entries from add_ids
            add_x = pd.DataFrame(add_x, index=add_x[:, 0]).loc[add_ids].to_numpy()
            add_y = pd.DataFrame(add_y, index=add_y[:, 0]).loc[add_ids].to_numpy()

        print('{file}: add set shapes:'.format(file=input_file), add_x.shape, add_y.shape)

        outfile = output_dir+input_file+'_filtered.npz'
        print('write to file: ' + outfile)
        np.savez_compressed(
            outfile,
            train_x=train_x,
            train_y=train_y,
            dev_x=dev_x,
            dev_y=dev_y,
            test_x=test_x,
            test_y=test_y,
            add_x=add_x,
            add_y=add_y
        )

    #####################################################
    # 3) resample train set                             #
    #    (apply resampling to top-down features,        #
    #     use the resulting ids for bottom-up features) #
    #####################################################

    colors = preprocessing.basic_colors()

    train_ids = np.array([])

    # begin with object_types, since only frequent objects are present in this set
    for file in ['object_types', 'histograms_bgr']:

        print('file:', file)

        import_arrays = np.load(input_dir+file+'_filtered.npz')

        train_x = import_arrays['train_x']
        train_y = import_arrays['train_y']

        if len(train_ids) == 0:

            train_y_df = pd.DataFrame(train_y[:, 1:], columns=colors, index=train_y[:, 0])
            train_y = pd.DataFrame(columns=colors)

            for c in colors:
                train_y = train_y.append(train_y_df.loc[train_y_df[c] == 1].sample(train_instances_per_color, replace=True, random_state=random_state))

            train_y = np.append(
                np.array(train_y.index).reshape(-1, 1),
                train_y.to_numpy(),
                axis=1
            )
            np.random.shuffle(train_y)

            train_x = pd.DataFrame(train_x[:, 1:], index=train_x[:,0]).loc[train_y[:,0]]
            train_x = np.append(
                np.array(train_x.index).reshape(-1, 1),
                train_x.to_numpy(),
                axis=1
            )

            train_ids = train_y[:,0]

        else:
            train_x_df = pd.DataFrame(train_x[:, 1:], index=train_x[:,0]).loc[train_ids]
            train_x = np.append(train_ids.reshape(-1, 1), train_x_df.to_numpy(), axis=1)
            train_y_df = pd.DataFrame(train_y[:, 1:], index=train_y[:,0]).loc[train_ids]
            train_y = np.append(train_ids.reshape(-1, 1), train_y_df.to_numpy(), axis=1)

        outfile = output_dir+file+'_resampled.npz'
        print('write to file: '+outfile)
        np.savez_compressed(
            outfile,
            train_x=train_x,
            train_y=train_y,
            dev_x=import_arrays['dev_x'],
            dev_y=import_arrays['dev_y'],
            test_x=import_arrays['test_x'],
            test_y=import_arrays['test_y'],
            add_x=import_arrays['add_x'],
            add_y=import_arrays['add_y']
        )
        print('Shape X:', train_x.shape)
        print('Shape Y:', train_y.shape)
    # save ids to file
    outfile = output_dir+'ids.npz'
    print('write ids to file: '+outfile)
    np.savez_compressed(
        outfile,
        train=train_ids,
        dev=import_arrays['dev_y'][:, 0],
        test=import_arrays['test_y'][:, 0],
        add=import_arrays['add_y'][:, 0]
    )
