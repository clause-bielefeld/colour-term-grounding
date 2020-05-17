import pandas as pd
import os
from ijson import items
import configparser
import sys
import re

sys.path.append(os.path.abspath('../utils'))
import preprocessing

config = configparser.ConfigParser()
config.read('../config.ini')

vg_json = config['PATHS']['vg-json']
vg_json_export = config['PATHS']['data']
image_dir = config['PATHS']['vg-images']

export_dir = vg_json_export + 'extracted_data/'

test_ratio = 0.1
dev_ratio = 0.2
train_ratio = 1 - (test_ratio + dev_ratio)


def split_name_list(row, new_df):

    image_id = row.image_id
    object_id = row.object_id
    name_list = row.object_name
    attribute = row.color
    bb_h = row.bb_h
    bb_w = row.bb_w
    bb_x = row.bb_x
    bb_y = row.bb_y

    for name in name_list:
        # print(image_id, object_id, name, attribute)
        new_df.append({
            'image_id': image_id,
            'object_id': object_id,
            'object_name': name.lower(),
            'color': attribute.lower(),
            'bb_h': bb_h,
            'bb_w': bb_w,
            'bb_x': bb_x,
            'bb_y': bb_y
        })  # , ignore_index=True)


def remove_colors_from_names(name, regexp):
    if re.match(regexp, name):
        return (re.sub(regexp, '', name))
    else:
        return (name)


# basic colour terms, cf. Berlin & Kay 1969
basic_colors = preprocessing.basic_colors()

if __name__ == "__main__":
    print('get objects with annotated colours')
    with open(vg_json+'attributes.json', 'r') as f:
        out = []
        for entry in items(f, 'item'):
            image_id = entry['image_id']
            for attributes in entry['attributes']:
                out_obj = {}
                object_id = attributes.get('object_id', None)
                object_name = attributes.get('names', None)
                w = attributes.get('w', None)
                h = attributes.get('h', None)
                x = attributes.get('x', None)
                y = attributes.get('y', None)
                if attributes.get('attributes', None) != None:
                    for attribute in attributes['attributes']:
                        if attribute.lower() in basic_colors:
                            out_obj = {
                                'image_id': image_id,
                                'object_id': object_id,
                                'object_name': object_name,
                                'color': attribute.lower(),
                                'bb_w': w,
                                'bb_h': h,
                                'bb_x': x,
                                'bb_y': y
                                }
                            out.append(out_obj)
                            out_obj = {}

    objects = pd.DataFrame.from_dict(out)

    # split entries if there are multiple object names
    objects_entries = []
    objects.apply(lambda x: split_name_list(x, objects_entries),axis=1)
    objects = pd.DataFrame.from_dict(objects_entries)
    print("df shape:", objects.shape)

    # remove colour terms from object names
    regexp = r'\b(black|white|red|green|yellow|blue|brown|orange|pink|purple|gray|grey)( |\b)'
    objects.object_name = objects.apply(lambda x: remove_colors_from_names(x.object_name, regexp), axis=1)

    if not os.path.isdir(export_dir):
        print('created dir {d}'.format(d=export_dir))
        os.mkdir(export_dir)

    # export as csv
    export = objects.to_csv(export_dir+"all_objects.csv")
