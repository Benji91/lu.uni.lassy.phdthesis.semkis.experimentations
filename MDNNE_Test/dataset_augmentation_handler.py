import utils as utils
import dataset_augmentator_lib as augmentator
import data_illustration_lib as datahandler


#AVERAGE IMAGE GENERATIONS


def generate_distored_images(number_of_generated_images, imagefolder, probability, grid_width, grid_height, magnitude):
    cleanFolder = "/formatted_average_images/" + imagefolder + "/output"
    filename = "formatted_" + imagefolder + ".png"
    directory = "formatted_average_images/" + imagefolder + "/"
    plot_fname = "augmented_" + imagefolder + ".png"
    utils.clean_result_folder(cleanFolder)
    distorted_images = augmentator.distortion(number_of_generated_images, directory, filename, probability=probability, grid_width=grid_width, grid_height=grid_height, magnitude=magnitude)
    datahandler.plot_set_of_images(distorted_images, filename=plot_fname)




imagefolder = "image_0"
number_of_generated_images = 16
generate_distored_images(number_of_generated_images, imagefolder, probability=1, grid_height=5, grid_width=5, magnitude=3)

imagefolder = "image_1"
number_of_generated_images = 16
generate_distored_images(number_of_generated_images, imagefolder, probability=1, grid_height=5, grid_width=5, magnitude=3)

imagefolder = "image_2"
number_of_generated_images = 16
generate_distored_images(number_of_generated_images, imagefolder, probability=1, grid_height=5, grid_width=5, magnitude=3)

imagefolder = "image_3"
number_of_generated_images = 16
generate_distored_images(number_of_generated_images, imagefolder, probability=1, grid_height=5, grid_width=5, magnitude=3)

imagefolder = "image_4"
number_of_generated_images = 16
generate_distored_images(number_of_generated_images, imagefolder, probability=1, grid_height=5, grid_width=5, magnitude=3)

imagefolder = "image_5"
number_of_generated_images = 16
generate_distored_images(number_of_generated_images, imagefolder, probability=1, grid_height=5, grid_width=5, magnitude=3)

imagefolder = "image_6"
number_of_generated_images = 16
generate_distored_images(number_of_generated_images, imagefolder, probability=1, grid_height=5, grid_width=5, magnitude=3)

imagefolder = "image_7"
number_of_generated_images = 16
generate_distored_images(number_of_generated_images, imagefolder, probability=1, grid_height=5, grid_width=5, magnitude=3)

imagefolder = "image_8"
number_of_generated_images = 16
generate_distored_images(number_of_generated_images, imagefolder, probability=1, grid_height=5, grid_width=5, magnitude=3)

imagefolder = "image_9"
number_of_generated_images = 16
generate_distored_images(number_of_generated_images, imagefolder, probability=1, grid_height=5, grid_width=5, magnitude=3)