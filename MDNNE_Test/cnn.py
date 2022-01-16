import os

from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.models import Sequential

import data_illustration_lib as datahandler
# from keras.optimizers import RMSprop
import utils as utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

utils.clean_result_folder()

x_train, y_train, x_test, y_test, x_val, y_val = datahandler.mnist_dataset_construction()

# ARCHITECTURE : CONVOLUTIONAL NEURAL NETWORK
# Archtecture In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
model = Sequential()

model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
# Define the optimizer
# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

#save untrained model to disk
utils.save_model(model, "not_trained_model")


# Set a learning rate annealer
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# Training configurations
epochs = 1
# Batch size defines number of samples that going to be propagated through the network.
batch_size = 84

# Train the neural network without data augmentation i obtained an accuracy of 0.98114
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), verbose=2,
                    shuffle=True)

#save trained model to disk
utils.save_model(model, "trained_model")

# PLOT THE HISTORY OF THE TRAINING (ACCURACY AND LOSS)
datahandler.plot_loss_and_accuracy_curve(history)

# PREDICTION ON THE VALIDATION DATASET
utils.generate_confustion_matrix(model, x_val, y_val, "validation")

# PREDICTION ON THE TEST DATASET
utils.generate_confustion_matrix(model, x_test, y_test, "testing")

# Evaluate the model on the test data set
datahandler.evaluate_model(model, x_test, y_test, "test dataset")
