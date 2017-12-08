import os
import numpy as np


def get_data_path(filename):
    path = os.path.join(
        os.path.dirname(__file__), os.pardir, "data", filename)
    if not os.path.exists(path):
        path = os.path.join(
            os.path.dirname(__file__), "data", filename)
    if not os.path.exists(path):
        path = os.path.join(
            os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        raise Exception("Could not find data file: {}".format(filename))

    return path


def get_data_and_monitor_digit_classification():
    mnist_path = get_data_path("mnist.npz")

    with np.load(mnist_path) as data:
        train_images = data["train_images"]
        train_labels = data["train_labels"]
        test_images = data["test_images"]
        test_labels = data["test_labels"]

    num_train = len(train_images)

    train_labels_one_hot = np.zeros((num_train, 10))
    train_labels_one_hot[range(num_train), train_labels] = 1
    return train_images, [train_labels, train_labels_one_hot], test_images, test_labels
