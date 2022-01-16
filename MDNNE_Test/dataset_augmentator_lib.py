import cv2
import numpy as np
import Augmentor
import glob
import data_illustration_lib as datahandler
from PIL import Image


root = "./results/"
p = Augmentor.Pipeline()


def plot_initial_image(parent_folder_path, filename) :
    init_file_path = parent_folder_path + filename
    image = datahandler.read_from_file(init_file_path)
    return image


def generate_images(parent_folder_path):
    images = []
    path_of_images = parent_folder_path + "output/*.png"
    filelist = glob.glob(path_of_images)
    for fname in filelist:
        array = datahandler.read_from_file(filename=fname)
        images.append(array)
    return images


def distortion(number_of_samples, path, filename, probability, grid_width, grid_height, magnitude):
    parent_folder_path = root+path
    images = []
    image = plot_initial_image(parent_folder_path, filename)
    images.append(image)

    p = Augmentor.Pipeline(parent_folder_path)
    p.random_distortion(probability=probability, grid_width=grid_width, grid_height=grid_height, magnitude=magnitude)
    p.sample(number_of_samples)
    generated_images = generate_images(parent_folder_path)

    merged_list = images + generated_images
    return merged_list


def rotate(number_of_samples, path, filename):
    parent_folder_path = root+path
    images = []
    image = plot_initial_image(parent_folder_path, filename)
    images.append(image)

    p = Augmentor.Pipeline(parent_folder_path)
    p.rotate(probability=1, max_left_rotation=180, max_right_rotation=180)
    p.sample(number_of_samples)
    generated_images = generate_images(parent_folder_path)

    merged_list = images + generated_images
    return merged_list


def translation(image, x_trans, y_trans):
    rows, cols = image.shape
    translation_matrix = np.float32([[1, 0, x_trans], [0, 1, y_trans]])
    trans_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return trans_image


def rotation(image, degrees):
    rows, cols = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), degrees, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image


def reformat(images):
    result = []
    for img in images :
        gray_image = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        gray_image = invert_colours(gray_image)
        result.append(gray_image)
    return result


def import_images(id):
    finalPath = 'images/'+str(id)+'/*.png'
    images = [cv2.imread(file) for file in glob.glob(finalPath)]
    return images


def image_flip(number_of_samples):
    p = Augmentor.Pipeline("./images")
    p.flip_random(probability=1)
    p.sample(number_of_samples)


def randomCrop():
    p = Augmentor.Pipeline("./images/0")
    p.crop_random(probability=1, percentage_area=0.8)
    g = p.keras_generator(batch_size=128)
    images, labels = next(g)
    return images, labels


def invert_colours(image):
    return 255 - image


