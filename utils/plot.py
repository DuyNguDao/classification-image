import warnings
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_cm(CM, normalize=True, save_dir='', names=(), show=True):
    """
    function: plot confusion matrix
    :param CM: array cm
    :param normalize: normaize 0-1
    :param save_dir: path save
    :param names: name class
    :param show: True
    :return:
    """
    if True:
        import seaborn as sn
        array = CM / ((CM.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
        if not normalize:
            array = np.asarray(array, dtype='int')
        fmt = 'd'
        if normalize:
            fmt = '.2f'
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        sn.set(font_scale=1.0 if 2 < 50 else 0.8)  # for label size
        labels = (0 < len(names) < 99) and len(names) == 2  # apply names to ticklabels
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array, annot=2 < 30, annot_kws={"size": 8}, cmap='Blues', fmt=fmt, square=True,
                       xticklabels=names if labels else "auto",
                       yticklabels=names if labels else "auto").set_facecolor((1, 1, 1))
        fig.axes[0].set_xlabel('True')
        fig.axes[0].set_ylabel('Predicted')
        if show:
            plt.show()
        name_save = 'confusion_matrix.png'
        if normalize:
            name_save = 'confusion_matrix_normalize.png'
        fig.savefig(Path(save_dir) / name_save, dpi=250)
        plt.close()


def plot_image(image, title=''):
    b, g, r = cv2.split(image)
    frame_rgb = cv2.merge((r, g, b))
    plt.imshow(frame_rgb)
    plt.title(title)
    plt.show()


def plot_and_export_image(list_predict_true, list_predict_false, numbers, class_name, save_dir=''):
    """
    Program Created by Dao Duy Ngu
    function: plot and save image min and max predict true, max predict fasle
    input: + list_predict_true: this is a list data include as: (score, label, image)
    + list_predict_false: ...
    + numbers: the numbers of image plot follow row
    + class_name: list of name class
    + save_dir: path save
    output: image save at path save_dir
    """
    list_plot = []
    for num_cls in range(len(class_name)):
        # add list follow class
        list_plot += list_predict_true[num_cls * 2 * numbers:(num_cls + 1) * 2 * numbers]
        list_plot += list_predict_false[num_cls * numbers:(num_cls + 1) * numbers]
        # initialize and plot
        fig = plt.figure(figsize=(13, 10), tight_layout=True)
        for num in range(1, 3 * numbers + 1):
            plt.subplot(3, numbers, num)
            plt.imshow(list_plot[num - 1][2])
            plt.xticks([]), plt.yticks([])
            if list_plot[num - 1][1] == class_name[num_cls]:
                plt.xlabel('True')
            else:
                plt.xlabel('False')
            plt.title(list_plot[num - 1][1] + ': %.2f%%' % (list_plot[num - 1][0]))
        # save plot
        fig.savefig(Path(save_dir) / (class_name[num_cls] + '.png'), dpi=400)
        plt.close()
        # clear list plot
        list_plot.clear()

