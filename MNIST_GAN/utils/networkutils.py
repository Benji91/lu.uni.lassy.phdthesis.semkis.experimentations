from keras.models import model_from_json
import numpy as np
import logging


def save_model(model, fullpath):
    """
    Generic Function for Saving a neural network model
    :param model:
    :param fullpath:
    :return:
    """
    # serialize networks to JSON
    model_json = model.to_json()
    with open(fullpath + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(fullpath + ".h5")
    logging.debug("Saved Neural Network to disk")


def load_model(fullpath):
    """
    Generic Function for Loading a Neural Network model
    :param fullpath:
    :return:
    """
    # load json and create networks
    json_file = open(fullpath + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new networks
    loaded_model.load_weights(fullpath + ".h5")
    logging.debug("Loaded Neural Network from disk")
    return loaded_model


def predict(model, x):
    """
    This function computes the outputs of the neural network and returns the result in the prediction and in one hot vector
    """
    # Predict the values from the validation da taset
    Y_pred = model.predict(x)
    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    print('Prediction done')
    return Y_pred, Y_pred_classes
