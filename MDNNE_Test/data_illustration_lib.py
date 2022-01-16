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

# GLOBAL VARIABLES
root = "./results/"
img_size = 28
img_shape = (img_size, img_size)


def read_from_file(filename):
    if root not in filename :
        path = root + filename
    else:
        path = filename
    x = Image.open(path).convert('L')
    x.thumbnail(img_shape)
    array = np.asarray(x.getdata(), dtype=np.float64).reshape((x.size[1], x.size[0]))
    np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x), threshold=np.nan)
    return array


def compute_average_of_images(x_train, y_train, label):
    number_of_images = len(x_train)
    average_image = np.zeros([28, 28], dtype=float)
    for row in range(28):
        for col in range(28):
            sum = 0
            count = 0
            for i in range(number_of_images):
                if int(y_train[i]) == label:
                    sum += int(x_train[i][row][col])
                    count += 1
            average_image[row][col] = sum / count
    filename = "average_image_" + str(label)
    average_image = np.around(average_image)
    save_image(average_image, directory="average_images", filename=filename)
    np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x), threshold=np.nan)


def extract_mnist_dataset():
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

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    return x_train, y_train, x_test, y_test


def mnist_dataset_construction():
    x_train, y_train, x_test, y_test = extract_mnist_dataset()

    random_seed = 2
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=random_seed)

    # Plot the input datasets for analysis purposes
    plot_bar_x(y_train, "Frequency of training images", "Number", "Frequencies")
    plot_bar_x(y_val, "Frequencies of validation images", "Number", "Frequencies")
    plot_bar_x(y_test, "Frequencies of testing images", "Number", "Frequencies")

    # The number of classes for the classification needed at the output
    num_classes = 10

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    return x_train, y_train, x_test, y_test, x_val, y_val


def plot_loss_and_accuracy_curve(history):
    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
    ax[1].legend(loc='best', shadow=True)
    path = root + "training_diagram"
    plt.savefig(path)
    plt.show()


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
    path = root + title
    plt.savefig(path)
    plt.show()


def predict(model, x):
    """
    This function computes the outputs of the neural network and returns the result in the prediction and in one hot vector
    """
    # Predict the values from the validation dataset
    Y_pred = model.predict(x)
    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    print('Prediction done')
    return Y_pred, Y_pred_classes


def plot_confusion_matrix(Y_expected, Y_predicted, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(Y_expected, Y_predicted)
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
    path = root + title
    plt.savefig(path)
    plt.show()
    classes = list(map(str, classes))
    print(classification_report(Y_expected, Y_predicted, target_names=classes))


def filter_errors(Y_pred_classes, Y_pred_probs, Y_expected, x_val):
    # Errors are difference between predicted labels and expected labels
    errors = (Y_pred_classes - Y_expected != 0)

    Y_pred_classes_errors = Y_pred_classes[errors]
    Y_expected_errors = Y_expected[errors]

    Y_pred_probs_errors = Y_pred_probs[errors]
    X_val_errors = x_val[errors]

    # Probabilities of the wrongly predicted numbers
    Y_prob_pred_wrong_numbers = np.max(Y_pred_probs_errors, axis=1)

    # Predicted probabilities of the expected values in the error set
    Y_prob_exp_numbers = np.diagonal(np.take(Y_pred_probs_errors, Y_expected_errors, axis=1))

    # Difference between the probability of the predicted label and the expected label
    delta_pred_true_errors = Y_prob_pred_wrong_numbers - Y_prob_exp_numbers

    # Sorted list of the delta prob errors indices (ordered in ascending order)
    sorted_delta_prob_errors = np.argsort(delta_pred_true_errors)
    return sorted_delta_prob_errors, X_val_errors, Y_pred_classes_errors, Y_expected_errors


def display_errors(img_errors, pred_errors, obs_errors, label):
    """ This function shows 6 images with their predicted and real labels"""
    dictionary = dict()
    for i in range(10):
        key = i
        values = []
        total_number_of_images = len(obs_errors)
        for label_index in range(total_number_of_images):
            if int(obs_errors[label_index]) == i:
                values.append(img_errors[label_index])
        dictionary[key] = values

    grid_size = 0
    for key, value in dictionary.items():
        # print value
        maximum = len(value)
        if (maximum > grid_size):
            grid_size = maximum

    print("Erroneous images from the ", label, " set :", len(img_errors))

    print(grid_size)
    fig, ax = plt.subplots(10, grid_size, figsize=(40.0, 40.0))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # print 10 examples for each 0-9 case
    for i in range(10):
        # find the value i in the first 500 images
        item_index = np.where(obs_errors == i)
        item_index = item_index[0]
        number_of_plots = len(item_index)
        for j in range(grid_size):
            if j < number_of_plots:
                im_index = item_index[j]
                ax[i, j].imshow(img_errors[im_index].reshape(img_shape), cmap='binary')
                ax[i, j].set_xlabel("exp : {} - pred: {}".format(obs_errors[im_index], pred_errors[im_index]))
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
            if j >= number_of_plots:
                fig.delaxes(ax[i, j])
    path = root + label
    plt.savefig(path)
    plt.show(block=False)


def plot_set_of_images(images, filename, figsize=(25,25)):
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
    path = root + "summary_synthetic_data/" + filename
    plt.savefig(path)
    plt.show(block=False)


def plot_average_and_formatted_images(original_images, formatted_images, label, nrows=10, ncols=2, figsize=(25,25)):
    """ This function shows 20 images in their real and formatted format
    """

    number_of_images = len(original_images)

    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i in range(number_of_images):
        ax[i, 0].imshow((original_images[i]).reshape(img_shape), cmap='gray')
        ax[i, 0].set_xlabel("Original")
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        ax[i, 1].imshow((formatted_images[i]).reshape(img_shape), cmap='gray')
        ax[i, 1].set_xlabel("Formatted")
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])

    path = root+"image_"+label
    plt.savefig(path)
    plt.show(block=False)


def save_image(image, directory, filename):

    my_dpi = 42
    plt.figure(figsize=(28/my_dpi, 28/my_dpi), dpi=my_dpi)
    plt.axis('off')
    plt.imshow(image, cmap="gray")
    directory_path = root + directory + "/"
    if not os.path.exists(directory_path):
       os.mkdir(directory_path)
    path = directory_path + filename
    plt.savefig(path, dpi=my_dpi)
    plt.show()


def display_image(image):
    """ This function shows 6 images with their predicted and real labels"""
    plt.imshow(image, cmap="gray")
    plt.show()


def evaluate_model(model, x_test, y_test, settype):
    print("Evaluation of Model on ", settype)
    score = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    print("%s: %.2f%%" % (model.metrics_names[0], score[0] * 100))


def merge_images(images):
    number_of_images = len(images)
    nrows = 28
    ncols = 28
    result = np.zeros([28, 28], dtype=float)
    for row in range(nrows):
        for col in range(ncols):
            max = 0
            for i in range(number_of_images):
                value = images[i][row][col]
                if value > max:
                    max = value
            result[row][col] = max
    return result
