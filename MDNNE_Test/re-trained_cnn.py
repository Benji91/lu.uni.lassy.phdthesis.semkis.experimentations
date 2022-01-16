import utils as utils
import os
from keras.models import Sequential
import data_illustration_lib as datahandler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#TODO: FIND A WAY TO RECONSTRUCT THE NEW AUGMENTED DATASET
x_train, y_train, x_test, y_test, x_val, y_val = datahandler.mnist_dataset_construction()
#ADD THE NEWLY CREATED IMAGES INSIDE THE ORIGINAL DATASET

#CODE HERE

model = Sequential()
model = utils.load_model("not_trained_model")
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])


# Training configurations
epochs = 30
# Batch size defines number of samples that going to be propagated through the network.
batch_size = 84

# Train the neural network without data augmentation i obtained an accuracy of 0.98114
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), verbose=2,
                    shuffle=True)

utils.save_model(model, "re_trained_model")

# PLOT THE HISTORY OF THE TRAINING (ACCURACY AND LOSS)
datahandler.plot_loss_and_accuracy_curve(history)

# PREDICTION ON THE VALIDATION DATASET
utils.generate_confustion_matrix(model, x_val, y_val, "validation")

# PREDICTION ON THE TEST DATASET
utils.generate_confustion_matrix(model, x_test, y_test, "testing")

# Evaluate the model on the test data set
datahandler.evaluate_model(model, x_test, y_test, "test dataset")