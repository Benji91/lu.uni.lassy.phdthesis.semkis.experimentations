import cv2
import math
import os

import matplotlib.pyplot as plt
import numpy as np

from utils import configutils

# GLOBAL VARIABLES
img_size = 28
img_shape = (img_size, img_size)


def generate_noise(shape: tuple):
    noise = np.random.uniform(0, 1, size=shape)
    return noise


def normalise_images(images: np.ndarray):
    """
    Normalise images to [-1, 1]
    """
    images = (images.astype(np.float32) - 127.5) / 127.5
    return images


def reverse_normalisation_images(images: np.ndarray):
    """
    From the [-1, 1] range transform the images back to [0, 255]
    """
    images = images * 127.5 + 127.5
    images = images.astype(np.uint8)
    return images


def generate_condition_embedding(label: int, nb_of_label_embeddings: int, number_of_conditions: int):
    label_embeddings = np.zeros((nb_of_label_embeddings, number_of_conditions))
    label_embeddings[:, label] = 1
    return label_embeddings


def plot_set_of_images(generator, number_of_conditions: int, filename, show_figure: bool, figsize=(25, 25)):
    images = []
    for i in range(10):
        noise = generate_noise((10, 100))
        label_input = generate_condition_embedding(i, 10, number_of_conditions)
        gen_images = generator.predict([noise, label_input], verbose=0)
        images.extend(gen_images)

    number_of_images = len(images)
    cols = math.sqrt(number_of_images)

    ncols = int(round(cols))
    if ncols*ncols <= number_of_images:
        ncols+1
    nrows = ncols

    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    counter = 0
    for i in range(ncols):
        for j in range(nrows):
            if counter < number_of_images:
                ax[i, j].imshow((images[counter]).reshape(img_shape), cmap='gray')
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
            if counter >= number_of_images:
                fig.delaxes(ax[i, j])
            counter += 1

    path = configutils.configSectionMap("FOLDERS")['synthetic_images_grids_path'] + filename
    plt.savefig(path)
    # plt.show(block=False)
    plt.clf()
    plt.cla()
    plt.close()


def save_generated_image(image, epoch, iteration, folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    file_path = "{0}/{1}_{2}.png".format(folder_path, epoch, iteration)
    cv2.imwrite(file_path, image.astype(np.uint8))


def show_progress(e, i, g0, d0, g1, d1):
    print("\repoch: %d, batch: %d, g_loss: %f, d_loss: %f, g_accuracy: %f, d_accuracy: %f" % (e, i, g0, d0, g1, d1))
    """sys.stdout.write(
        "\repoch: %d, batch: %d, g_loss: %f, d_loss: %f, g_accuracy: %f, d_accuracy: %f" % (e, i, g0, d0, g1, d1))
    sys.stdout.flush()
    """