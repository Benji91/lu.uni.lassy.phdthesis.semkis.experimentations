from utils import configutils, networkutils, plotutils, datautils
import matplotlib.pyplot as plt
import numpy as np
import logging



def plot_cnn_training_curve(history):
    """
     Plotting a CNN Training Curve
    :param history: Training History
    :return:
    """
    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
    ax[1].legend(loc='best', shadow=True)
    path = configutils.configSectionMap("FOLDERS")['cnn_training_path'] + configutils.configSectionMap("CNN")['cnn_training_performance']
    plt.savefig(path)
    plt.show()
    message = "Cnn Training Curve has been stored @" + path
    logging.debug(msg=message)


def generate_confusion_matrix(model, x, y, label):
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
    Y_pred, Y_pred_classes = networkutils.predict(model, x)

    np.save(configutils.configSectionMap("FOLDERS")['dataset_path'] + label + "_" + configutils.configSectionMap("DATA")['y_pred'],
            Y_pred)
    np.save(configutils.configSectionMap("FOLDERS")['dataset_path'] + label + "_" + configutils.configSectionMap("DATA")['y_pred_classes'],
            Y_pred_classes)

    # Convert validation observations to one hot vectors
    Y_expected = np.argmax(y, axis=1)

    # plot the confusion matrix
    cm_title = "Confusion Matrix - " + label + " data"
    plotutils.plot_confusion_matrix(Y_expected, Y_pred_classes,
                                    classes=range(10),
                                    title=cm_title)

    # DISPLAY THE ERROR RESULTS ON THE VALIDATION DATASET
    sorted_delta_prob_errors, X_errors, Y_pred_classes_errors, Y_true_errors, Y_pred_probs_errors = datautils.filter_errors(
        Y_pred_classes, Y_pred, Y_expected, x)

    np.save(configutils.configSectionMap("FOLDERS")['dataset_path'] + label + "_" + configutils.configSectionMap("DATA")[
            'x_errors'],X_errors)
    np.save(configutils.configSectionMap("FOLDERS")['dataset_path'] + label + "_" + configutils.configSectionMap("DATA")[
            'y_errors'], Y_pred_probs_errors)
    # sorted indices of the output values
    sorted_predicted_output_indices = np.argsort(-Y_pred_probs_errors)
    # Sorted probabilities
    sorted_Y_probs_errors = Y_pred_probs_errors[np.arange(np.shape(Y_pred_probs_errors)[0])[:,np.newaxis], sorted_predicted_output_indices]

    # Display the errors the errors
    plotutils.plot_errors(X_errors, Y_pred_classes_errors, Y_true_errors, sorted_Y_probs_errors, sorted_predicted_output_indices, label)
