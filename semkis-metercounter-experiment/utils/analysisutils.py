from utils import configutils, networkutils, plotutils, datautils
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.io import save
from bokeh.models import Title

import matplotlib.pyplot as plt
import numpy as np
import logging, skimage, random


def generate_ds_2digitshift(images, labels):
    syn_images = []
    syn_labels = []
    for image, label in zip(images, labels):
        if label[10] == 1 or label[19] == 1:
            syn_images.append(image)
            syn_labels.append(label)

    syn_images = np.array(syn_images)
    syn_labels = np.array(syn_labels)
    return syn_images, syn_labels


def generateSetOfNoisyImages(test_imgs, test_labels):
    noisy_images = []
    noisy_labels = []
    for img, label in zip(test_imgs, test_labels):
        noisyImageNumber = np.random.randint(low=0, high=6)
        for _ in range(noisyImageNumber):
            amount = random.uniform(0.1, 0.9)
            salt_vs_pepper = random.uniform(0.1, 0.9)
            noisy_images.append(generateNoise(img, "s&p", amount=amount, salt_vs_pepper=salt_vs_pepper))
            noisy_labels.append(label)

    noisy_images = np.array(noisy_images)
    noisy_labels = np.array(noisy_labels)
    return noisy_images, noisy_labels


def generateNoise(img, mode, amount, salt_vs_pepper):
    return skimage.util.random_noise(img, mode=mode, amount=amount, salt_vs_pepper= salt_vs_pepper)


def plot_web_all_data(x, y, filename):
    # bokeh library tests
    plots_images = []
    children_images = []

    d1_class, d2_class, y_classes = filter_outputs(y, N=2)

    # indices = np.argsort(y_classes)
    images = x
    y_classes = y_classes

    for img, label, d1, d2 in zip(images, y_classes, d1_class, d2_class):
        if len(children_images) == 10:
            plots_images.append(children_images)
            children_images = []

        p = figure(plot_width=400, plot_height=400, title="Class:" + adapt_labels(label))
        p.x_range.range_padding = p.y_range.range_padding = 0
        p.add_layout(Title(text="L-D:" + str(d1), align="center"), "above")
        p.add_layout(Title(text="R-D:" + str(d2), align="center"), "above")
        p.add_layout(Title(text="CLA:" + str(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])), align="center"), "above")
        d = np.flipud(img * 255)
        p.image(image=[d], x=0, y=0, dw=400, dh=400)
        p.grid.grid_line_width = 0.5

        children_images.append(p)

    if 0 < len(children_images):
        plots_images.append(children_images)

    grid1 = gridplot(plots_images, plot_width=300, plot_height=300)
    filename = "../results/analysis/cnn/data/" + filename + "_images.html"
    save(grid1, filename=filename)


def generate_web_analystics(x, y, y_predicted, round):
    # bokeh library tests
    plots_cr_images = []
    children_cr_images = []
    plots_ncr_images = []
    children_ncr_images = []

    d1_class, d2_class, y_classes = filter_outputs(y, N=2)
    d1_class, d2_class, y_pred_classes = filter_outputs(y_predicted, N=2)
    y_pred_rounded = np.around(y_predicted, decimals=1)

    indices = np.argsort(y_classes)
    images = x[indices]
    y_classes = y_classes[indices]
    y_pred_classes = y_pred_classes[indices]
    y_pred_rounded = y_pred_rounded[indices]

    for img, label, pred_label, prediction in zip(images, y_classes, y_pred_classes, y_pred_rounded):
        if len(children_cr_images) == 10:
            plots_cr_images.append(children_cr_images)
            children_cr_images = []

        if len(children_ncr_images) == 10:
            plots_ncr_images.append(children_ncr_images)
            children_ncr_images = []

        p = figure(plot_width=400, plot_height=400,
                   title="Class:" + adapt_labels(label) + " - Reco:" + adapt_labels(pred_label))
        p.x_range.range_padding = p.y_range.range_padding = 0

        p.add_layout(Title(text="R-D:" + str(prediction[:len(prediction) // 2]), align="center"), "above")
        p.add_layout(Title(text="L-D:" + str(prediction[:len(prediction) // 2]), align="center"), "above")
        p.add_layout(Title(text="CLA:" + str(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])), align="center"), "above")

        d = np.flipud(img * 255)
        p.image(image=[d], x=0, y=0, dw=400, dh=400)
        p.grid.grid_line_width = 0.5

        if label != pred_label:
            children_ncr_images.append(p)
        else:
            children_cr_images.append(p)

    if 0 < len(children_cr_images):
        plots_cr_images.append(children_cr_images)

    if 0 < len(children_ncr_images):
        plots_ncr_images.append(children_ncr_images)

    grid1 = gridplot(plots_ncr_images, plot_width=300, plot_height=300)
    filename = "../results/analysis/cnn/data/r" + str(round) + "_ncr_images.html"
    save(grid1, filename=filename)

    grid2 = gridplot(plots_cr_images, plot_width=300, plot_height=300)
    filename = "../results/analysis/cnn/data/r" + str(round) + "_cr_images.html"
    save(grid2, filename=filename)


def adapt_labels(label):
    if 0 <= label < 10:
        return "0" + str(label)
    else:
        return str(label)


def plot_cnn_training_curve(history):
    """
     Plotting a CNN Training Curve
    :param history: Training History
    :return:
    """
    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="Development loss", axes=ax[0])
    ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_accuracy'], color='r', label="Development accuracy")
    ax[1].legend(loc='best', shadow=True)
    path = configutils.configSectionMap("FOLDERS")['cnn_training_path'] + configutils.configSectionMap("CNN")[
        'cnn_training_performance']
    plt.savefig(path)
    plt.show()
    message = f"CNN Training Curve has been stored @{path}"
    print(message)
    logging.debug(msg=message)


def filter_outputs(y, N):
    if N == 1:
        return np.argmax(y, axis=1)

    indicesone, indicestwo = np.split(y, 2, axis=1)

    first_digit_classes = np.argmax(indicesone, axis=1)
    second_digit_classes = np.argmax(indicestwo, axis=1)
    combined_digit_classes = first_digit_classes * 10 + second_digit_classes
    return first_digit_classes, second_digit_classes, combined_digit_classes


def generate_confusion_matrix(model, x, y, label, num_classes, N):
    """
    Generation of a Confusion Matrix
    :param model: Sequential
    :param x: Input Dataset
    :param y: Output Dataset
    :param label: Type of Set
    :return:
    """

    # PREDICTION ON THE VALIDATION DATASET
    # Predict the values from the validation dataset
    y_pred, y_pred_classes = networkutils.predict(model, x)

    print(y_pred.shape)
    print(y_pred_classes.shape)
    # Convert validation observations to one hot vectors
    # TODO: Change these two lines depening on context
    y_d1_expected, y_d2_expected, y_expected = filter_outputs(y, N)
    y_d1_predicted, y_d2_predicted, y_pred_classes = filter_outputs(y_pred, N)

    # plot the confusion matrix
    cm_title = "Confusion Matrix - {label} data"
    finallabel = label + "_all"
    plotutils.plot_confusion_matrix(
        y_expected, y_pred_classes,
        classes=num_classes,
        label=finallabel,
    )

    finallabel = label + "_d1"
    plotutils.plot_confusion_matrix(
        y_d1_expected, y_d1_predicted,
        classes=10,
        label=finallabel,
    )

    finallabel = label + "_d2"
    plotutils.plot_confusion_matrix(
        y_d2_expected, y_d2_predicted,
        classes=10,
        label=finallabel,
    )
    """
    dataset_path = configutils.configSectionMap("FOLDERS")['dataset_path']
    config_data = configutils.configSectionMap("DATA")

    np.save(dataset_path + label + "_" + config_data['y_pred'], y_pred)
    np.save(dataset_path + label + "_" + config_data['y_pred_classes'], y_pred_classes)


    # DISPLAY THE ERROR RESULTS ON THE VALIDATION DATASET
    sorted_delta_prob_errors, x_errors, y_pred_classes_errors, y_true_errors, y_pred_probs_errors = datautils.filter_errors(
        y_pred_classes, y_pred, y_expected, x
    )

    np.save(dataset_path + label + "_" + config_data['x_errors'], x_errors)
    np.save(dataset_path + label + "_" + config_data['y_errors'], y_pred_probs_errors)

    # sorted indices of the output values
    sorted_predicted_output_indices = np.argsort(-y_pred_probs_errors)

    # Sorted probabilities
    sorted_y_probs_errors = y_pred_probs_errors[
        np.arange(np.shape(y_pred_probs_errors)[0])[:, np.newaxis], sorted_predicted_output_indices
    ]

    # Display the errors.
    plotutils.plot_errors(
        x_errors, y_pred_classes_errors, y_true_errors, sorted_y_probs_errors,
        sorted_predicted_output_indices, label
    )
    """
