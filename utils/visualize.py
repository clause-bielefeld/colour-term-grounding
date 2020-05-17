import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_vg_image(row, img_dir, size=5, edgecolor='green'):

    filename = img_dir + str(row.image_id)+'.jpg'
    image = cv2.imread(filename)

    # get bb
    bb = {
        'h': row.bb_h,
        'w': row.bb_w,
        'x': row.bb_x,
        'y': row.bb_y,
        'label': row.color+' '+row.object_name
    }

    # plot image
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(np.flip(image, axis=2))
    plt.text(
        bb['x']+bb['w']+10,
        bb['y']+bb['h']+10,
        bb['label'],
        fontsize=12,
        color='red',
        bbox=dict(facecolor='white')
        )

    # add bb
    rect = patches.Rectangle(
        (bb['x'], bb['y']), bb['w'], bb['h'],
        linewidth=3, edgecolor=edgecolor,
        facecolor='green', alpha=1, fill=False
        )
    ax.add_patch(rect)

    plt.show()

    return (plt)
