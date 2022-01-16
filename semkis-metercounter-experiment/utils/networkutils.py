from keras.models import model_from_json
import numpy as np
import logging


def save_model(model, full_path):
    """
    Generic Function for Saving a neural network model
    :param model:
    :param full_path:
    :return:
    """

    # Serialize networks to JSON
    model_json = model.to_json()
    with open(f"{full_path}.json", 'w') as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    model.save_weights(f"{full_path}.h5")
    logging.debug("Saved Neural Network to disk")


def load_model(full_path):
    """
    Generic Function for Loading a Neural Network model
    :param full_path:
    :return:
    """
    # Load json and create networks
    json_file = open(f"{full_path}.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # Load weights into new networks
    loaded_model.load_weights(f"{full_path}.h5")
    logging.debug("Loaded Neural Network from disk")
    return loaded_model


def predict(model, x):
    """
    Compute the outputs of the neural network.
    :return: The result in the prediction and in one hot vector
    """
    # Predict the values from the validation dataset
    y_pred = model.predict(x)

    # Convert predictions classes to one hot vectors
    y_pred_classes = np.argmax(y_pred, axis=1)
    print("Prediction done")
    return y_pred, y_pred_classes
