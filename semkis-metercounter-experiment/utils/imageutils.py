import math

import matplotlib.pyplot as plt
import numpy as np


def show_images(images):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    print("Show image")

    number_of_images = len(images)
    x = int(math.ceil(math.sqrt(number_of_images)))
    print("Number of images : ", x)

    count = 0

    fig, ax = plt.subplots(x, x, figsize=(80.0, 80.0))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for i in range(x):
        for j in range(x):
            if count < number_of_images:
                ax[i, j].imshow(images[count].reshape((28, 28)), cmap='gray')
            if count >= number_of_images:
                fig.delaxes(ax[i, j])
            count += 1
    plt.show(block=False)
    plt.clf()
    plt.cla()
    plt.close()
    print("Show Image End")


def combine_images(generated_images):
    total, width, height = generated_images.shape[:-1]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total) / cols)
    combined_image = np.zeros((height * rows, width * cols),
                              dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index / cols)
        j = index % cols
        combined_image[width * i:width * (i + 1), height * j:height * (j + 1)] = image[:, :, 0]
    return combined_image


def display_image(image):
    """ This function shows 6 images with their predicted and real labels"""
    plt.imshow(image, cmap="gray")
    plt.show()
