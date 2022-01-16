from networks.cdcgan import DCGAN
from networks.cnn import CNN
from keras.datasets import mnist
import keras.utils as kerasutils
from utils import configutils, datautils, plotutils, imageutils
import numpy as np
import matplotlib.pyplot as plt
import math



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
    print(x)

    count = 0

    fig, ax = plt.subplots(x, x, figsize=(80.0, 80.0))
    #fig.subplots_adjust(hspace=0.5, wspace=0.5)
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


configutils.initDirectories()

# Only used once to build the datasets and to store it
# ======================
x_train, y_train, x_test, y_test, x_dev, y_dev = datautils.mnist_dataset_construction()

# Used to load the datasets and store them in the array
x_train = np.load(configutils.configSectionMap("FOLDERS")['dataset_path'] + configutils.configSectionMap("DATA")['x_train'] + ".npy")
y_train = np.load(configutils.configSectionMap("FOLDERS")['dataset_path'] + configutils.configSectionMap("DATA")['y_train'] + ".npy")
x_test = np.load(configutils.configSectionMap("FOLDERS")['dataset_path'] + configutils.configSectionMap("DATA")['x_test'] + ".npy")
y_test = np.load(configutils.configSectionMap("FOLDERS")['dataset_path'] + configutils.configSectionMap("DATA")['y_test'] + ".npy")
x_dev = np.load(configutils.configSectionMap("FOLDERS")['dataset_path'] + configutils.configSectionMap("DATA")['x_dev'] + ".npy")
y_dev = np.load(configutils.configSectionMap("FOLDERS")['dataset_path'] + configutils.configSectionMap("DATA")['y_dev'] + ".npy")

#filter = np.where(y_dev == 1)
#x_dev = x_dev[filter]
#y_dev = y_dev[filter]

#show_images(x_dev)
# Engineering of initial CNN and training it
# ======================
cnn = CNN(x_train, y_train, x_test, y_test, x_dev, y_dev)
cnn.build_cnn()

"""
cnn = CNN(x_train, y_train, x_test, y_test, x_dev, y_dev)
cnn.build_cnn()
history = cnn.train(epoch=50)
cnn.generate_confusion_matrices()
cnn.evaluate_model()
"""
"""
# Load TRAINED CNN and evaluate the model
# ======================
cnn_trained = CNN(x_train, y_train, x_test, y_test, x_dev, y_dev)
cnn_trained.build_trained_cnn()
"""

"""
# Load UNTRAINED CNN and evaluate the model
# ======================
cnn_untrained = CNN(x_train, y_train, x_test, y_test, x_dev, y_dev)
cnn_untrained.build_untrained_cnn()

#Only needed if we want to retrain the untrained model! Trained Model is saved!
#cnn_untrained.train(filename="")
#cnn_untrained.evaluate_model()
#cnn_untrained.generate_confusion_matrices()


# build cdcgan
# ======================
print("Start Training")
cdcgan = DCGAN()

# epoch = 0
#cdcgan.load_gan_for_epoch(epoch=epoch)
#cdcgan.train(quantity=50, labels=[7], min_nb_pdiff=0, max_nb_pdiff=50, threshold=150, min_epoch=epoch)

cdcgan.load_generator(epoch=340)

number_of_images = 10000
images, labels = cdcgan.generate_images(number_of_images=number_of_images, label=7)

# Filter images that produced errornuous data
cnn_untrained.x_train = np.append(cnn_untrained.x_train, images, axis=0)
cnn_untrained.y_train = np.append(cnn_untrained.y_train,  labels, axis=0)

cnn_untrained.train(filename="retrained_"+str(number_of_images)+"_")
cnn_untrained.evaluate_model(filename="retrained_"+str(number_of_images)+"_")
cnn_untrained.generate_confusion_matrices()
"""