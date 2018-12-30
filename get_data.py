import random
import cv2
import os
import pickle
import numpy as np


# transform images into matrix format
def get_training_data():
    global DATADIR
    global CATEGORIES
    global training_data

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)    # get gray-scale image
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:  # corrupted images will be passed
                pass


if __name__ == "__main__":
    DATADIR = 'C:\\python\\machine learning\\tf\\PetImages' # change to the directory where you store your images
    CATEGORIES = ['Dog', 'Cat']
    training_data = []
    IMG_SIZE = 80   # change to the image size you wish to compress to. Keep it small to reduce training complexity
    get_training_data()
    random.shuffle(training_data)   # shuffle our training data
    sample_num = len(training_data)

    # check cat and dog distribution. If it is close to 0.5 then it should be fine
    img_sum = []
    for sample in training_data[int(sample_num * 0.8):]:
        img_sum.append(sample[1])
    print('image sum = {}'.format(np.mean(img_sum)))

    # initialize training, validation and testing data. If need validation data, then try ratio of 70:15:15 for
    # training, validation and testing data sample respectively. Otherwise, try 80:20 for training and testing
    # data sample
    X_train = []
    y_train = []
    # X_val = []
    # y_val = []
    X_test = []
    y_test = []

    # append matrix and label into training data
    for feature, label in training_data[:int(sample_num * 0.8)]:
        X_train.append(feature)
        y_train.append(label)

    # for feature, label in training_data[int(sample_num * 0.7):int(sample_num * 0.85)]:
    #     X_val.append(feature)
    #     y_val.append(label)

    for feature, label in training_data[int(sample_num * 0.8):]:
        X_test.append(feature)
        y_test.append(label)

    # reshape image data by adding an extra level since maxpooling2D function requires 4D training data
    X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # generate training and testing data
    pickle_out = open("X_train.pickle", "wb")
    pickle.dump(X_train, pickle_out)
    pickle_out.close()

    pickle_out = open("y_train.pickle", "wb")
    pickle.dump(y_train, pickle_out)
    pickle_out.close()

    # pickle_out = open("X_val.pickle", "wb")
    # pickle.dump(X_val, pickle_out)
    # pickle_out.close()
    #
    # pickle_out = open("y_val.pickle", "wb")
    # pickle.dump(y_val, pickle_out)
    # pickle_out.close()

    pickle_out = open("X_test.pickle", "wb")
    pickle.dump(X_test, pickle_out)
    pickle_out.close()

    pickle_out = open("y_test.pickle", "wb")
    pickle.dump(y_test, pickle_out)
    pickle_out.close()
