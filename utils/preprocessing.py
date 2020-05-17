import cv2
import numpy as np
from scipy.stats import entropy
from math import log


class out():
    x = np.empty((0, 513))
    x_temp = np.empty((0, 513))
    y = np.empty((0, 12))
    y_temp = np.empty((0, 12))


class status():
    entries_count = 0


def basic_colors():
    # cf. Berlin & Kay 1969
    return ([
        'black', 'blue', 'brown', 'gray',
        'green', 'orange', 'pink', 'purple',
        'red', 'white', 'yellow'
        ])


def calculate_entropy(row, normalize=False, columns=basic_colors()):
    if normalize:
        return (entropy([row[c] for c in columns]) / log(len(columns)))
    return entropy([row[c] for c in columns])


def get_histogram(filename, bb, img_dir, verbose=True, convert=False):
    # https://docs.opencv.org/master/d8/dbc/tutorial_histogram_calculation.html
    # import image, set img dimensions and pixel count
    ##################################################
    image = cv2.imread(filename)
    h, w, x, y = bb
    nbins = 8

    # convert bgr image to lab, hsv or ycc
    if convert == 'lab':
        if verbose == 'all':
            print('converting to L*a*b* color space')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        ranges = [0, 256, 0, 256, 0, 256]
        # https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
    elif convert == 'hsv':
        if verbose == 'all':
            print('converting to HSV color space')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ranges = [0, 180, 0, 256, 0, 256]
        # https://docs.opencv.org/3.4.3/df/d9d/tutorial_py_colorspaces.html
    else:
        ranges = [0, 256, 0, 256, 0, 256]

    img_height = image.shape[0]
    img_width = image.shape[1]
    pixel = img_width * img_height

    # create mask for bb
    ####################
    # bb corner coordinates
    p1 = (x, y)
    p2 = (x+w, y+h)

    # np array with zeros and the same shape as img
    mask = np.zeros(image.shape[:2], dtype="uint8")
    # set values to 1 which correspond to bb
    cv2.rectangle(mask, p1, p2, (1, 1, 1), -1)

    # create 3d histogram
    #####################
    hist = cv2.calcHist([image], [0, 1, 2],
        mask, [nbins, nbins, nbins], ranges)

    # relative instead of absolute frequencies
    rel_hist = np.divide(hist, pixel)
    # ravel histogram
    rel_hist = rel_hist.ravel()

    return rel_hist


def histogram_from_series(row, img_dir, verbose=True, convert=False):
    # get values from df row
    #########################
    entry_id = row['index']
    image_id = row.image_id

    h = row.bb_h
    w = row.bb_w
    x = row.bb_x
    y = row.bb_y

    filename = img_dir+str(image_id)+'.jpg'

    rel_hist = get_histogram(
        filename=filename, bb=[h, w, x, y], img_dir=img_dir,
        verbose=verbose, convert=convert
        )

    # create output
    #################
    # output arrays (first entry is id for both arrays)
    output_x = np.insert(rel_hist, 0, entry_id)
    output_y = np.array(([entry_id, row.color_black, row.color_blue,
                                    row.color_brown, row.color_gray,
                                    row.color_green, row.color_orange,
                                    row.color_pink, row.color_purple,
                                    row.color_red, row.color_white,
                                    row.color_yellow]))
    # stack output arrays to out.x and out.y
    out.x_temp = np.vstack((out.x_temp,output_x))
    out.y_temp = np.vstack((out.y_temp,output_y))

    if len(out.x_temp) >= 5000:

        if verbose:
            print('cspace: {cspace} {entry_number} / {entry_count} : {entry_id}'.format(
                cspace=convert,
                entry_number=row.name + 1,
                entry_count=status.entries_count,
                entry_id=entry_id
                ))

        out.x = np.vstack((out.x, out.x_temp))
        out.y = np.vstack((out.y, out.y_temp))
        out.x_temp = np.empty((0, 513))
        out.y_temp = np.empty((0, 12))


def x_y_histograms(df, img_dir, verbose=True, convert=False):

    print('calculating histograms, verbose: {v}, converting: {c}'.format(v=verbose,c=convert))
    status.entries_count = len(df)

    out.x = np.empty((0, 513))
    out.y = np.empty((0, 12))
    out.x_temp = np.empty((0, 513))
    out.y_temp = np.empty((0, 12))

    df.apply(
        lambda x: histogram_from_series(
                x, img_dir, verbose=verbose, convert=convert
            ), axis=1
        )

    out.x = np.vstack((out.x, out.x_temp))
    out.y = np.vstack((out.y, out.y_temp))
    return(out.x, out.y)


def no_color_in_name(name, color_list):
    if any(color in name for color in color_list):
        return False
    else:
        return True


def freq_cdo_cno(
        df, num_cdos=100, num_cnos=100, num_cbos=100,
        min_num=100, return_entropy=False
        ):
    """
    return frequent objects, color diagnostic objects (+ associated color), color neutral objects
    """
    colors = basic_colors()
    object_names = df.groupby('object_name').size().reset_index(name='count')
    object_names = object_names.loc[object_names['count'] >= min_num]
    frequent_objects = list(object_names.object_name)

    colors_per_object = df\
        .drop(['image_id', 'object_id', 'bb_x', 'bb_y', 'bb_w', 'bb_h'], axis=1)\
        .pivot_table(index='object_name',
                   columns='color',
                   aggfunc=len,
                   fill_value=0)

    colors_per_object = colors_per_object[
        colors_per_object.index.isin(frequent_objects)
        ]
    colors_per_object = colors_per_object[
        colors_per_object.index.map(
                lambda x: no_color_in_name(x, colors)
            ).values
        ]

    # calculate entropy
    colors_per_object['entropy'] = colors_per_object.apply(
            lambda x: calculate_entropy(x), axis=1
        )

    # return dict: objects with lowest entropy + most probable colour
    color_diagnostic_objects = colors_per_object.sort_values('entropy')

    if num_cdos:
        color_diagnostic_objects = color_diagnostic_objects.iloc[:num_cdos]
    color_diagnostic_objects = color_diagnostic_objects.idxmax(axis=1).to_dict()
    # return list: objects with highest entropy
    color_neutral_objects = colors_per_object.sort_values('entropy', ascending=False)

    if num_cnos:
        color_neutral_objects = color_neutral_objects.iloc[:num_cnos]
    color_neutral_objects = color_neutral_objects.index.tolist()

    color_biased_objects = colors_per_object.sort_values('entropy', ascending=False)

    if num_cbos:
        middle = round(len(colors_per_object)/2)
        lower_b = middle - round(num_cbos/2)
        upper_b = middle + num_cbos - round(num_cbos/2)
        color_biased_objects = color_biased_objects.iloc[lower_b:upper_b]
    color_biased_objects = color_biased_objects.index.tolist()

    if return_entropy:
        objects_entropy = colors_per_object['entropy'].to_dict()
        return (
            frequent_objects, color_diagnostic_objects, color_biased_objects,
            color_neutral_objects, objects_entropy
            )
    else:
        return (
            frequent_objects, color_diagnostic_objects,
            color_biased_objects, color_neutral_objects
            )
