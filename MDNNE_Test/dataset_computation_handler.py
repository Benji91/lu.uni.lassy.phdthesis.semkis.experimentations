import data_illustration_lib as datahandler
from keras.datasets import mnist
import numpy as np
import utils as utils
import dataset_augmentator_lib as augmentator
(x_train, y_train), (x_test, y_test) = mnist.load_data()

size = 28,28
print(x_train[0][0][0])
x_train = 255-x_train
x_test = 255-x_test
print(x_train[0][0][0])

original_images = []
formatted_images = []

# Computing the average of all images
datahandler.compute_average_of_images(x_train, y_train, 0)
datahandler.compute_average_of_images(x_train, y_train, 1)
datahandler.compute_average_of_images(x_train, y_train, 2)
datahandler.compute_average_of_images(x_train, y_train, 3)
datahandler.compute_average_of_images(x_train, y_train, 4)
datahandler.compute_average_of_images(x_train, y_train, 5)
datahandler.compute_average_of_images(x_train, y_train, 6)
datahandler.compute_average_of_images(x_train, y_train, 7)
datahandler.compute_average_of_images(x_train, y_train, 8)
datahandler.compute_average_of_images(x_train, y_train, 9)

#AVERAGE IMAGE CLEANER
average_image = datahandler.read_from_file("average_images/average_image_0.png")
original_images.append(average_image)
formatted_image = np.copy(average_image)
formatted_image[formatted_image > 100] = 255
formatted_image[formatted_image <= 100] = 0
formatted_images.append(formatted_image)
datahandler.save_image(formatted_image, directory="formatted_average_images/image_0", filename="formatted_image_0.png")

average_image = datahandler.read_from_file("average_images/average_image_1.png")
original_images.append(average_image)
formatted_image = np.copy(average_image)
formatted_image[formatted_image > 150] = 255
formatted_image[formatted_image <= 130] = 0
formatted_images.append(formatted_image)
datahandler.save_image(formatted_image, directory="formatted_average_images/image_1", filename= "formatted_image_1.png")


average_image = datahandler.read_from_file("average_images/average_image_2.png")
original_images.append(average_image)
formatted_image = np.copy(average_image)
formatted_image[formatted_image > 80] = 255
formatted_image[formatted_image <= 80] = 0
formatted_images.append(formatted_image)
datahandler.save_image(formatted_image, directory="formatted_average_images/image_2", filename= "formatted_image_2.png")


average_image = datahandler.read_from_file("average_images/average_image_3.png")
original_images.append(average_image)
formatted_image = np.copy(average_image)
formatted_image[formatted_image > 100] = 255
formatted_image[formatted_image <= 100] = 0
formatted_images.append(formatted_image)
datahandler.save_image(formatted_image, directory="formatted_average_images/image_3", filename= "formatted_image_3.png")

average_image = datahandler.read_from_file("average_images/average_image_4.png")
original_images.append(average_image)
formatted_image = np.copy(average_image)
formatted_image[formatted_image > 170] = 255
formatted_image[formatted_image <= 170] = 0
formatted_images.append(formatted_image)
datahandler.save_image(formatted_image, directory="formatted_average_images/image_4", filename= "formatted_image_4.png")

average_image = datahandler.read_from_file("average_images/average_image_5.png")
original_images.append(average_image)
formatted_image = np.copy(average_image)
formatted_image[formatted_image > 120] = 255
formatted_image[formatted_image <= 120] = 0
formatted_images.append(formatted_image)
datahandler.save_image(formatted_image, directory="formatted_average_images/image_5", filename= "formatted_image_5.png")

average_image = datahandler.read_from_file("average_images/average_image_6.png")
original_images.append(average_image)
formatted_image = np.copy(average_image)
formatted_image[formatted_image > 130] = 255
formatted_image[formatted_image <= 130] = 0
formatted_images.append(formatted_image)
datahandler.save_image(formatted_image, directory="formatted_average_images/image_6", filename= "formatted_image_6.png")

average_image = datahandler.read_from_file("average_images/average_image_7.png")
original_images.append(average_image)
formatted_image = np.copy(average_image)
formatted_image[formatted_image > 110] = 255
formatted_image[formatted_image <= 110] = 0
formatted_images.append(formatted_image)
datahandler.save_image(formatted_image, directory="formatted_average_images/image_7", filename= "formatted_image_7.png")

average_image = datahandler.read_from_file("average_images/average_image_8.png")
original_images.append(average_image)
formatted_image = np.copy(average_image)
formatted_image[formatted_image > 130] = 255
formatted_image[formatted_image <= 130] = 0
formatted_images.append(formatted_image)
datahandler.save_image(formatted_image, directory="formatted_average_images/image_8", filename= "formatted_image_8.png")

average_image = datahandler.read_from_file("average_images/average_image_9.png")
original_images.append(average_image)
formatted_image = np.copy(average_image)
formatted_image[formatted_image > 180] = 255
formatted_image[formatted_image <= 180] = 0
formatted_images.append(formatted_image)
datahandler.save_image(formatted_image, directory="formatted_average_images/image_9", filename= "formatted_image_9.png")

# print(len(original_images))
# print(len(formatted_images))
# datahandler.plot_average_and_formatted_images(original_images, formatted_images, label="average_images")
