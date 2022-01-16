from keras.utils import to_categorical
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import logging
import numpy as np
from utils import configutils, cdcgan_utils, plotutils
from skimage.measure import compare_ssim as ssim


def mean_squared_error(imageA, imageB):
    err = np.sum((imageA - imageB) ** 2)
    err /= float(imageA.shape[0]+ imageA.shape[1])
    return err


def similarity(first_image, second_image, min_nb_pdiff, max_nb_pdiff, threshold):
    diff = first_image - second_image
    diff = np.absolute(diff)
    nb_diff_pixels = diff[(diff > threshold)].size

    if nb_diff_pixels > max_nb_pdiff:
        return 0
    else:
        if nb_diff_pixels <= min_nb_pdiff:
            return 1
        else:
            return 1 - nb_diff_pixels / max_nb_pdiff


def quality_of_image(generated_image, ref_dataset, min_nb_pdiff, max_nb_pdiff, threshold):
    mse_similarities = []
    ssim_similarities = []
    similarities = []

    for image in ref_dataset :
        mse_similarities.append(mean_squared_error(generated_image, image))
        similarities.append(similarity(generated_image, image, min_nb_pdiff, max_nb_pdiff, threshold))
        ssim_similarities.append(ssim(generated_image, image))
    return np.amax(np.array(similarities)), np.amin(np.array(mse_similarities)), np.amax(np.array(ssim_similarities))


def quality_of_generator(generator, quantity, labels, min_nb_pdiff, max_nb_pdiff, threshold):
    mse_similarities = []
    ssim_similarities = []
    similarities = []
    (x_train, y_train), (_, _) = mnist.load_data()

    for label in labels:
        train_filter = np.where(y_train == label)
        ref_dataset = x_train[train_filter]
        n = 1000  # for 1000 random indices
        index = np.random.choice(ref_dataset.shape[0], n, replace=False)
        ref_dataset = ref_dataset[index]
        for i in range(quantity):
            noise = cdcgan_utils.generate_noise((1, 100))
            label_input = cdcgan_utils.generate_condition_embedding(label, 10, 10)
            gen_image = generator.predict([noise, label_input], verbose=0)
            pixels = cdcgan_utils.reverse_normalisation_images(gen_image)
            pixels = pixels.reshape((28, 28))
            csim, mse, ssim = quality_of_image(pixels, ref_dataset, min_nb_pdiff, max_nb_pdiff, threshold)
            similarities.append(csim)
            mse_similarities.append(mse)
            ssim_similarities.append(ssim)
    return np.average(np.array(similarities)), np.average(np.array(mse_similarities)), np.average(np.array(ssim_similarities))


def mnist_dataset_construction():
    """
    Building the MNIST DataSet
    :return:
    """
    x_train, y_train, x_test, y_test = restructure_mnist_dataset()

    random_seed = 2
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=random_seed)

    # Plot the input datasets for analysis purposes
    plotutils.plot_bar_x(y_train, "Frequency of training images", "Number", "Frequencies")
    plotutils.plot_bar_x(y_val, "Frequencies of validation images", "Number", "Frequencies")
    plotutils.plot_bar_x(y_test, "Frequencies of testing images", "Number", "Frequencies")

    # The number of classes for the classification needed at the output
    num_classes = 10

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    y_val = to_categorical(y_val, num_classes)

    np.save(configutils.configSectionMap("FOLDERS")['dataset_path'] + configutils.configSectionMap("DATA")['x_train'],
            x_train)
    np.save(configutils.configSectionMap("FOLDERS")['dataset_path'] + configutils.configSectionMap("DATA")['y_train'],
            y_train)
    np.save(configutils.configSectionMap("FOLDERS")['dataset_path'] + configutils.configSectionMap("DATA")['x_dev'],
            x_val)
    np.save(configutils.configSectionMap("FOLDERS")['dataset_path'] + configutils.configSectionMap("DATA")['y_dev'],
            y_val)
    np.save(configutils.configSectionMap("FOLDERS")['dataset_path'] + configutils.configSectionMap("DATA")['x_test'],
            x_test)
    np.save(configutils.configSectionMap("FOLDERS")['dataset_path'] + configutils.configSectionMap("DATA")['y_test'],
            y_test)
    return x_train, y_train, x_test, y_test, x_val, y_val


def restructure_mnist_dataset():
    """
    Restructuring the MNIST dataset
    :return:
    """
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # data normalisation
    x_train /= 255
    x_test /= 255

    # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    logging.debug('x_train shape:', x_train.shape)
    logging.debug('y_train shape:', y_train.shape)
    logging.debug('x_test shape:', x_test.shape)
    logging.debug('y_test shape:', y_test.shape)
    logging.debug(x_train.shape[0], 'train samples')
    logging.debug(x_test.shape[0], 'test samples')

    return x_train, y_train, x_test, y_test


def filter_errors(Y_pred_classes, Y_pred_probs, Y_expected, x_val):
    """
    Function for filtering the errournous predictions
    :param Y_pred_classes: All predicted output classes
    :param Y_pred_probs: All predicted probability vectors
    :param Y_expected: All expected output classes
    :param x_val:
    :return:
    """
    # Errors are defined by the difference between predicted output and expected output
    errors = (Y_pred_classes - Y_expected != 0)

    # Filter the errors for the predicted and expected output
    Y_pred_classes_errors = Y_pred_classes[errors]
    Y_expected_errors = Y_expected[errors]

    # Filter the errors on predicted probabilties and input values
    Y_pred_probs_errors = Y_pred_probs[errors]
    X_val_errors = x_val[errors]

    # PREDICTED VALUES : Probabilities of the wrongly predicted handwritten digits
    Y_prob_pred_wrong_numbers = np.max(Y_pred_probs_errors, axis=1)

    # EXPECTED VALUES : Predicted probabilities of the expected values in the error set
    Y_prob_exp_numbers = np.diagonal(np.take(Y_pred_probs_errors, Y_expected_errors, axis=1))

    # Difference between the probability of the predicted label and the expected label
    delta_pred_true_errors = Y_prob_pred_wrong_numbers - Y_prob_exp_numbers

    # Sorted list of the delta prob errors indices (ordered in ascending order)
    sorted_delta_prob_errors = np.argsort(delta_pred_true_errors)
    return sorted_delta_prob_errors, X_val_errors, Y_pred_classes_errors, Y_expected_errors, Y_pred_probs_errors
