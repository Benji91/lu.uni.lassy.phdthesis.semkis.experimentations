import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import logging
from utils import cdcgan_utils, configutils
import math, csv

img_size = 28
img_shape = (img_size, img_size)


def plot_confusion_matrix(y_expected, y_predicted, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_expected, y_predicted)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Expected label')
    plt.xlabel('Predicted label')
    path = configutils.configSectionMap("FOLDERS")['cnn_confusion_path'] + title
    plt.savefig(path)
    plt.show()
    classes = list(map(str, classes))
    logging.debug("Successfully Generated Confusion Matrix")
    logging.debug(classification_report(y_expected, y_predicted, target_names=classes))


def plot_training_statistics(d_loss_logs, g_loss_logs, d_accuracy_logs, g_accuracy_logs):
    d_loss_logs_a = np.array(d_loss_logs)
    g_loss_logs_a = np.array(g_loss_logs)
    d_accuracy_logs_a = np.array(d_accuracy_logs)
    g_accuracy_logs_a = np.array(g_accuracy_logs)
    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(d_accuracy_logs_a[:, 0], d_accuracy_logs_a[:, 1], label="Discriminator Accuracy")
    ax[0].plot(g_accuracy_logs_a[:, 0], g_accuracy_logs_a[:, 1], label="Generator Accuracy")
    ax[0].legend(loc='best', shadow=True)

    ax[1].plot(g_loss_logs_a[:, 0], g_loss_logs_a[:, 1], label="Generator Loss")
    ax[1].plot(d_loss_logs_a[:, 0], d_loss_logs_a[:, 1], label="Discriminator Loss")
    ax[1].legend(loc='best', shadow=True)

    plt.savefig(configutils.configSectionMap("FOLDERS")['dcgan_training_path'] +
                configutils.configSectionMap("DCGAN")['dcgan_performance_loss'])
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


def plot_generator_quality(computed_accuracy):
    computed_accuracy_a = np.array(computed_accuracy)
    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(computed_accuracy_a[:, 0], computed_accuracy_a[:, 1], label="Custom Similarity")
    ax[0].legend(loc='best', shadow=True)

    ax[1].plot(computed_accuracy_a[:, 0], computed_accuracy_a[:, 2], label="Mean-Squared Error Similarity")
    ax[1].legend(loc='best', shadow=True)

    ax[2].plot(computed_accuracy_a[:, 0], computed_accuracy_a[:, 3], label="Structural Similarity Measurey")
    ax[2].legend(loc='best', shadow=True)

    plt.savefig(configutils.configSectionMap("FOLDERS")['dcgan_training_path'] +
                configutils.configSectionMap("DCGAN")['dcgan_generator_performance'])
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


def plot_bar_x(data, title, x_label, y_label):
    # this is for plotting purpose
    data_frequencies = np.bincount(data)
    index = np.arange(len(data_frequencies))
    plt.bar(index, data_frequencies)
    print(title, ": ", data_frequencies)
    plt.xlabel(y_label, fontsize=10)
    plt.ylabel(x_label, fontsize=10)
    plt.xticks(index, index, fontsize=10, rotation=30)
    plt.title(title, fontsize=15)
    for a, b in zip(index, data_frequencies):
        plt.text(a, b, str(b), ha='center', va='bottom')
    path = configutils.configSectionMap("FOLDERS")['cnn_data_analysis_path'] + title
    plt.savefig(path)
    plt.show()


def plot_errors(images, class_predictions, expectations, sorted_predictions, sorted_prediction_indices, filename):
    """
    Function to plot the errors and generate a csv file of the incorrectly predicted images
    :param images: The list of incorrectly predicted images
    :param class_predictions: list of incorrectly predicted classes
    :param expectations: list of expected values
    :param sorted_predictions: sorted list of predicted probabilities
    :param sorted_prediction_indices: list of indices of the sorted image
    :param filename: name of the generated file
    :return:
    """

    number_of_classes = 10

    sorted_probabilities = np.array(list(zip(sorted_prediction_indices, sorted_predictions)))

    dictionary = dict()
    for i in range(number_of_classes):
        filter = np.where(expectations == i)
        filtered_images = images[filter]
        filtered_class_predictions = class_predictions[filter]
        filtered_expectations = expectations[filter]
        filtered_sorted_probabilities = sorted_probabilities[filter]
        dictionary[i] = [filtered_images, filtered_class_predictions, filtered_expectations,
                         filtered_sorted_probabilities]

    # Determine grid width
    grid_size = 0
    for key, value in dictionary.items():
        maximum = len(value[0])
        if maximum > grid_size:
            grid_size = maximum

    print("Erroneous images from the ", filename, " set :", len(images))

    print(grid_size)
    fig, ax = plt.subplots(number_of_classes, grid_size, figsize=(80.0, 80.0))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    single_image_data = []

    csv_data = [["ID", "PREDICTION", "EXPECTATION", "PREDICTION PER CLASS"]]
    count = 0
    for i in range(10):
        data = dictionary[i]
        number_of_plots = len(data[0])
        for j in range(grid_size):
            if j < number_of_plots:
                ax[i, j].imshow(data[0][j].reshape(img_shape), cmap='gray')
                ax[i, j].set_xlabel("id : {}".format(count), fontsize=80)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                sub_data = "["
                for k in range(len(data[3][j][0])):
                    sub_data += "(" + str(int(data[3][j][0][k])) + " ; " + str(round(data[3][j][1][k], 3)) + ") "
                sub_data += "]"

                single_image_data.append([count, data[0][j], data[1][j], data[2][j], str(round(data[3][j][1][0], 3))])
                csv_data.append([count, data[1][j], data[2][j], sub_data])
                count += 1
            if j >= number_of_plots:
                fig.delaxes(ax[i, j])

    path = configutils.configSectionMap("FOLDERS")['cnn_data_analysis_path'] + "images_errors_" + filename
    plt.savefig(path)
    plt.clf()
    plt.cla()
    plt.close()
    for d in single_image_data:
        plot_single_image(d[1], str(d[0])+"-"+str(d[2])+"-"+str(d[3])+"-"+str(d[4])+".png",
                          "ID= "+str(d[0]) + "  & PRED= " + str(d[2]) + " & EXP= " + str(d[3]) + " & PROB= " + str(d[4]),
                          filename)



    with open(configutils.configSectionMap("FOLDERS")['cnn_data_analysis_path'] + filename + ".csv", "w+") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csv_data)
    csvFile.close()


def plot_single_image(image, filename, label, settype):
    pixels = image.reshape((28, 28))
    """
    # The rest of columns are pixels
    pixels = cdcgan_utils.reverse_normalisation_images(pixels)

    # Reshape the array into 28 x 28 array (2-dimensional array)
    pixels = pixels.reshape((28, 28))
    """
    # Plot
    plt.title(label)
    plt.imshow(pixels, cmap='gray')
    path = configutils.configSectionMap("FOLDERS")['cnn_data_analysis_path']+settype+"_"+filename
    plt.savefig(path)
    plt.clf()
    plt.cla()
    plt.close()

def plot_set_of_images(images, filename, figsize=(25, 25)):
    number_of_images = len(images)
    cols = math.sqrt(number_of_images)

    ncols = int(round(cols))
    if ncols * ncols <= number_of_images:
        ncols + 1
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
    # TODO : DEFINE SAVE AREA for plotting set of images
    # path = root + "summary_synthetic_data/" + filename
    # plt.savefig(path)
    plt.show(block=False)
