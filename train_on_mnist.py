import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from dataset import get_data_and_monitor_digit_classification
from kmeans import Kmeans


def display_digit(digit, labeled = True, title = ""):
    """
    graphically displays a 784x1 vector, representing a digit
    """
    if labeled:
        digit = digit[1]
    image = digit
    plt.figure()
    fig = plt.imshow(image.reshape(28,28))
    fig.set_cmap('gray_r')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if title != "":
        plt.title("Inferred label: " + str(title))


train_images, [train_labels, train_labels_one_hot], test_images, test_labels = \
    get_data_and_monitor_digit_classification()

k = len(np.unique(train_labels))

k_means = Kmeans(train_images, k=k)

k_means.run()
