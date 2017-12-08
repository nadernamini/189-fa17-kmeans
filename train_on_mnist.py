import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from dataset import get_data_and_monitor_digit_classification
from kmeans import Kmeans


def display_digit(digit, title=""):
    """
    graphically displays a 784x1 vector, representing a digit
    """
    plt.figure()
    fig = plt.imshow(digit.reshape(28, 28))
    fig.set_cmap('gray_r')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if title != "":
        plt.title("Inferred label: " + str(title))
    plt.show()


train_images, [train_labels, train_labels_one_hot], test_images, test_labels = \
    get_data_and_monitor_digit_classification()

k = len(np.unique(train_labels))

print(test_images.shape)
k_means = Kmeans(train_images, train_labels, k=k)

k_means.run()

num_tests = 1

idx = np.random.choice(test_images.shape[0], num_tests, replace=False)
labels = k_means.classify(test_images[idx])
for j, id in enumerate(idx):
    display_digit(test_images[id], labels[j])
