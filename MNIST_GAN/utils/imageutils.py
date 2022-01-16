import matplotlib.pyplot as plt
import numpy as np
import more_itertools as itertools
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
import keras
from PIL import Image
import os
import math
import sys


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
