import utils as utils
import os
from keras.models import Sequential
import data_illustration_lib as datahandler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_train, y_train, x_test, y_test, x_val, y_val = datahandler.mnist_dataset_construction()
model = Sequential()
model = utils.load_model("trained_model")
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

# PREDICTION ON THE VALIDATION DATASET
utils.generate_confustion_matrix(model, x_val, y_val, "validation")

# PREDICTION ON THE TEST DATASET
utils.generate_confustion_matrix(model, x_test, y_test, "testing")

#Evaluate the model on the test data set
datahandler.evaluate_model(model, x_test, y_test, "test dataset")