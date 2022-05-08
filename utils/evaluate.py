from glob import glob
import cv2
from sklearn import metrics
from utils.plot import plot_cm, plot_and_export_image
import numpy as np


def evaluate_cm(model, test_set_path, save_cm='', normalize=True, show=True):
    """
    function: compute confusion matrix
    :param model: model predict
    :param test_set_path: path image predict image
    :param save_cm: path save
    :param normalize: normalize or no
    :param show: show confusion matrix
    :return:
    """
    truth_label = []
    pred_label = []
    for cls in model.class_name:
        list_img = glob(test_set_path + '/' + cls + '/*.jpg')
        for i in list_img:
            image = cv2.imread(i)
            pred = model.predict(image)
            truth_label.append(model.class_name_id[cls])
            pred_label.append(model.class_name_id[pred[0]])
    CM = metrics.confusion_matrix(truth_label, pred_label)
    # compute accuracy, precision, recall, f1-score
    f1_score_average = 0
    precision_average = 0
    recall_average = 0
    num_class = CM.shape[0]
    for i in range(num_class):
        TP = CM.T[i, i]
        FP = np.sum(CM.T[i, :]) - TP
        FN = np.sum(CM.T[:, i]) - TP
        if TP + FP + FN == 0:
            continue
        if num_class == 2:
            num_class = 1
        precision_average += TP / ((TP + FP) * num_class)
        recall_average += TP / ((TP + FN) * num_class)
        f1_score_average += TP / ((TP + (FP + FN) / 2) * num_class)
        if num_class == 1:
            break
    eye = np.eye(CM.shape[0])
    accuracy = np.sum(CM.T*eye)/np.sum(CM)
    print('Precision: ', round(precision_average, 2))
    print('Recall: ', round(recall_average, 2))
    print('F1-score: ', round(f1_score_average, 2))
    print('Accuracy: ', round(accuracy, 2))
    # plot confusion matrix
    plot_cm(CM.T, save_dir=save_cm, names=model.class_name, normalize=normalize, show=show)


def export_min_max(model, test_set_path, save_ep='', numbers=5):
    """
    Program Created by Dao Duy Ngu
    function: get numbers image predict have score max and min and plot image
    """
    # list predict true and false of a list image
    list_predict_true = []
    list_predict_false = []
    # list predict min_max predict true and max predict false limits numbers
    list_min_max_true = []
    list_max_false = []
    # list contain array image
    list_array_image = []
    count = 0
    for cls in model.class_name:
        # get path image follow class name
        list_image = glob(test_set_path + '/' + cls + '/*.jpg')
        # read and predict
        for path_image in list_image:
            # read image
            image = cv2.imread(path_image)
            # predict image
            predict = model.predict(image)
            # convert color bgr channel -> rgb
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # append image
            list_array_image.append(image)
            # check class predict and true
            if predict[0] == cls:
                list_predict_true.append((max(predict[1]), predict[0], count))
            else:
                list_predict_false.append((max(predict[1]), predict[0], count))
            count += 1
        # sort score predict min -> max
        list_predict_true = sorted(list_predict_true)
        list_predict_false = sorted(list_predict_false)
        # get numbers image predict true and flase have score max and min
        list_min_max_true += list_predict_true[0:numbers]
        list_min_max_true += list_predict_true[len(list_predict_true)-numbers:len(list_predict_true)]
        list_max_false += list_predict_false[len(list_predict_false)-numbers:len(list_predict_false)]
        # clear list predict
        list_predict_true.clear()
        list_predict_false.clear()
    # change index image and array image
    list_min_max_true = [(data[0], data[1], list_array_image[data[2]]) for data in list_min_max_true]
    list_max_false = [(data[0], data[1], list_array_image[data[2]]) for data in list_max_false]
    # plot and save
    plot_and_export_image(list_min_max_true, list_max_false, numbers=numbers,
                          class_name=model.class_name, save_dir=save_ep)


