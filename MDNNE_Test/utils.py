# let's keep our keras backend tensorflow quiet
from keras.models import model_from_json
import numpy as np
import data_illustration_lib as datahandler
import os

# GLOBAL VARIABLES
folder = "./results"


def clean_result_folder(path=""):
    directory = folder + path
    for the_file in os.listdir(directory):
        file_path = os.path.join(directory, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def save_model(model, filename):
    # serialize model to JSON
    model_json = model.to_json()
    with open("./results/" + filename + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./results/" + filename + ".h5")
    print("Saved model to disk")


def load_model(filename):
    # load json and create model
    json_file = open('./results/' + filename + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./results/" + filename + ".h5")
    print("Loaded model from disk")
    return loaded_model


def generate_confustion_matrix(model, x, y, label):
    # PREDICTION ON THE VALIDATION DATASET
    # Predict the values from the validation dataset
    Y_pred, Y_pred_classes = datahandler.predict(model, x)
    # Convert validation observations to one hot vectors
    Y_expected = np.argmax(y, axis=1)
    # plot the confusion matrix
    cm_title = "Confusion Matrix - " + label + " data"
    datahandler.plot_confusion_matrix(Y_expected, Y_pred_classes, classes=range(10),
                                      title=cm_title)

    # DISPLAY THE ERROR RESULTS ON THE VALIDATION DATASET
    sorted_delta_prob_errors, X_errors, Y_pred_classes_errors, Y_true_errors = datahandler.filter_errors(
        Y_pred_classes, Y_pred, Y_expected, x)
    # Show the errors
    filename = "images_errors_" + label
    datahandler.display_errors(X_errors, Y_pred_classes_errors, Y_true_errors, filename)
