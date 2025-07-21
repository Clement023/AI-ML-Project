

import cv2
import os
import numpy as np
import CNN
import preprocess


def main():
    print('OpenCV version {} '.format(cv2.__version__))

    curr_dir = os.path.dirname(__file__)

    a0 = '55'
    a1 = '31'
    training_folder = os.path.join(cur_dir, '/Training/',a0)
    test_folder = os.path.join(curr_dir, '/Test/', a1)

    training_data = []
    for filename in os.listdir(training_folder):
        img = cv2.imread(os.path.join(training_folder, filename), 0)
        if img is not None:
            data = np.array(preprocess.preparation(img))
            data = np.reshape(data, (901, 1))
            result = [[0], [1]] if "original" in filename else [[1], [0]]
            result = np.array(result)
            result = np.reshape(result, (2, 1))
            training_data.append((data, result))

    test_data = []
    for filename in os.listdir(test_folder):
        img = cv2.imread(os.path.join(test_folder, filename), 0)
        if img is not None:
            data = np.array(preprocess.preparation(img))
            data = np.reshape(data, (901, 1))
            result = 1 if "original" in filename else 0
            test_data.append((data, result))

    model = CNN.CNN([901, 500, 500, 2])
    model.sgd(training_data, 10, 50, 0.01, test_data)



if __name__ == '__main__':
    main()






