"""
Member: DA0 DUY NGU, LE VAN THIEN
"""

from glob import glob
import cv2
import argparse
import os
from utils.load_model import Model


def detect_image(path_image, path_model):
    """
    function: detect face mask of folder image
    :param path_image: path of folder contain image
    :return: None
    """
    if not os.path.exists(path_image):
        raise ValueError("Input folder (", path_image, ") does not exist.")
    list_images = glob(path_image + '/*.jpg') + glob(path_image + '/*.jpeg') + glob(path_image + '/*.png')
    list_images.sort()
    print("Number image: ", len(list_images))
    # load model face mask classification
    model = Model(path_model)

    for path in list_images:
        image = cv2.imread(path)
        pred = model.predict(image)
        cv2.putText(image, f'{pred[0]}: {round(max(pred[1]))}', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(f'{pred[0]}: {round(max(pred[1]))}', image)
        cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification image')
    parser.add_argument("--file-folder", help="folder file image", default='', type=str)
    parser.add_argument("--folder-model", help="folder contain file model", default='', type=str)
    args = parser.parse_args()
    path_image = args.file_folder
    path_model = args.folder_train
    detect_image(path_image, path_model)




