from PIL import Image
import numpy as np
import glob, logging, os, random, platform
from utils import configutils, plotutils
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from skimage import exposure
import skimage

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

number_of_references = 10
width_ref_img = 165
height_ref_img = 310


def shuffle_set(dataset, labels):
    """
    Shuffles randomly a given set
    :param dataset:
    :return:
    """
    indices = np.arange(dataset.shape[0])
    np.random.shuffle(indices)
    shuffled_dataset = dataset[indices]
    shuffled_labels = labels[indices]
    print("-------------------------")
    print("Images Shape : ", shuffled_dataset.shape)
    print("Label Shape : ", shuffled_labels.shape)
    print("-------------------------")

    return shuffled_dataset, shuffled_labels


def collect_classified_data(path):
    """
    Collects images from a certain path. In this case, we collect the reference images!
    :param path:
    :return:
    """
    images = []
    labels = []
    for f in glob.iglob(path + "*"):
        img = Image.open(f)
        file_location = img.filename
        operating_system = platform.system()
        if (operating_system == "Windows"):
            filename = file_location.split('\\')[-1]
        else:
            filename = file_location.split('/')[-1]
        filename = filename.split('.')
        print(filename)
        images.append(np.asarray(img))
        labels.append([filename[0]])

    # Transformation to Numpy Array
    images = np.array(images)[:, :, :, 1]
    labels = np.array(labels)
    images = images.reshape(10, 310, 165, 1)

    # Change the type of each entry
    images = images.astype('float32')
    labels = labels.astype('float32')

    # Sorting the array
    indices = np.argsort(labels, axis=0)
    labels = labels[indices]
    images = images[indices]

    # Normalisation of the Pixels to values in between 0 and 1
    images /= 255

    # Transform output categories to 10 possible values
    labels = to_categorical(labels, 10)

    print(images[0].shape, 'image shape')
    print(labels[0].shape, 'labels shape')

    print(images.shape[0], 'train samples')
    return images, labels


def load_set_from_file(filename):
    dataset_storage_path = configutils.configSectionMap("FOLDERS")['dataset_path']
    return np.load(dataset_storage_path + filename + ".npy")


def save_set_to_file(filename, data):
    """
    Local Saving of a Numpy Array to the Dataset Results Folder
    :param filename: name of the output file without extension
    :param data: Numpy Array containing all necessary data
    :return: empty
    """
    dataset_storage_path = configutils.configSectionMap("FOLDERS")['dataset_path']
    np.save(dataset_storage_path + filename, data)


def concatenate_images(firstset, secondset):
    """
    Concatenates two 2D same-sized images long the vertical axis
    :param firstimage: 2D Numpy Array of size (X,Y)
    :param secondimage: 2D Numpy Array of size (X,Y)
    :return: Concatenated 2D Numpy Array of size (2*X, Y)
    """
    return np.concatenate((firstset.reshape(310, 165), secondset.reshape(310, 165)), axis=1).reshape(310, 330, 1)


def concatenate_classes(firstset, secondset):
    """
    Concatenates two 2D same-sized images long the vertical axis
    :param firstimage: 2D Numpy Array of size (X,Y)
    :param secondimage: 2D Numpy Array of size (X,Y)
    :return: Concatenated 2D Numpy Array of size (2*X, Y)
    """
    return np.concatenate((firstset.reshape(1, 10), secondset.reshape(1, 10)), axis=1).reshape(20)


def generate_concatenated_images(images, labels):
    """
    Generates all possible combinations of two concatenated images from this input set
    :param images: Set of same sized 2D Numpy Array
    :return: Numpy Array of 2D Numpy Arrays
    """
    print("Reference Images Shape", images.shape)
    print("Reference Labels Shape", labels.shape)

    number_of_refimages = images.shape[0]
    metercounter_images = []
    image_A_classes = []
    image_B_classes = []
    for i in range(number_of_refimages):
        for j in range(number_of_refimages):
            metercounter_images.append(concatenate_images(images[i], images[j]))
            image_A_classes.append(labels[i])
            image_B_classes.append(labels[j])

    image_A_classes = np.array(image_A_classes)
    image_B_classes = np.array(image_B_classes)

    metercounter_images = np.array(metercounter_images)
    equivalence_classes = concatenate_classes(image_A_classes, image_B_classes)

    print("Concatenated Reference Images Shape", metercounter_images.shape)
    print("Classes Shape", equivalence_classes.shape)

    return metercounter_images, equivalence_classes


def generate_rangeof_shifted_images(images, labels, parameter_range, step, threshold):
    """
    Generate a Range of Shifted images
    :param images:
    :param labels:
    :param parameter_range:
    :param step:
    :param threshold:
    :return:
    """
    # Validation of the Parameters
    if not assert_parameters(parameter_range):
        print("Parameters are not valid!")
        return

    shifted_ref_images = []
    equivalence_classes = []

    counter = 0
    while counter < len(images):
        current_img = images[counter]
        current_label = labels[counter]

        if counter == len(images) - 1:
            next_img = images[0]
            next_label = labels[0]

        else:
            next_img = images[counter + 1]
            next_label = labels[counter + 1]

        shifted_ref_images.append(current_img)
        equivalence_classes.append(current_label)

        parameters = np.arange(parameter_range[0], parameter_range[1], step)
        for parameter in parameters:
            para_image, para_label = generate_shifted_images(current_img, next_img, current_label, next_label,
                                                             parameter, threshold)
            shifted_ref_images.append(para_image.reshape(310, 165, 1))
            equivalence_classes.append(para_label.reshape(10))
            parameter += step
        counter += 1

    # Transform to numpy array
    shifted_ref_images = np.array(shifted_ref_images)
    equivalence_classes = np.array(equivalence_classes)
    print("Reference Mechanical Image Shape : ", str(shifted_ref_images.shape))
    print("Reference Mechanical Classes Shape : ", str(equivalence_classes.shape))

    """
    # Retrieve filenames for mechanical images
    mec_ref_img_filename = configutils.configSectionMap("DATA")['mec_ref_images']
    mec_ref_labels_filename = configutils.configSectionMap("DATA")['mec_ref_labels']

    # Save mechanical labels to file
    save_set_to_file(mec_ref_img_filename, shifted_ref_images)
    save_set_to_file(mec_ref_labels_filename, equivalence_classes)
    
    plotutils.plot_set_of_images(shifted_ref_images, "lol", len(parameters)+1, figsize=(310, 165))
    """
    return shifted_ref_images, equivalence_classes


def assert_parameters(parameter_range):
    """
    Validating the parameters for mechanical image generation
    :param parameter_range:
    :return:
    """
    return parameter_range[0] > 0 or parameter_range[1] < 1


def generate_shifted_images(current_image, next_image, current_label, next_label, parameter, threshold):
    """
    Generation of specifically parametrised images
    :param images:
    :param labels:
    :param parameter:
    :param threshold:
    :return:
    """

    shift = int(height_ref_img * parameter)
    image_zero = np.copy(current_image.reshape(310, 165))
    image_one = np.copy(next_image.reshape(310, 165))
    if parameter <= threshold:
        equivalence_class = current_label
    else:
        equivalence_class = next_label

    image_zero2one = np.concatenate((image_zero[shift:310, :], image_one[0:shift, :]), axis=0)
    return image_zero2one.reshape(310, 165, 1), equivalence_class.reshape(10)


def plotnoise(img, mode, title):
    if mode is not None:
        gimg = skimage.util.random_noise(img, mode=mode)
        plt.imshow(gimg, cmap='gray')
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.show()


def plotbrightness(img, title, brightness_range, brightness_step):
    for parameter in np.arange(brightness_range[0], brightness_range[1], brightness_step):
        print(parameter)
        images = np.copy(img)
        images *= 255
        images = np.where((255 - images) < 100, 255, images + parameter)
        images /= 255
        plt.imshow(images, cmap='gray')
        plt.title("Brightness: " + str(parameter))
        plt.axis("off")
        plt.show()
    return


def generate_countermeter_dataset(ref_images, ref_labels, parameter_range, step, threshold):
    """
    Function used for generating mechanical and nonmechanical images used in first experiment
    Generates a complete set of a parametrised meter counter from 00 to 99...
    :param ref_images:
    :param ref_labels:
    :return:
    """

    # Validation of the Parameters
    if not assert_parameters(parameter_range):
        print("Parameters are not valid!")
        return

    # print("Parameters are valid! Building dataset... Please wait!")
    # print("Reference Images Shape : ", ref_images.shape)
    # print("Reference Labels Shape : ", ref_labels.shape)
    number_of_reference_images = ref_images.shape[0]

    # print("Number of Reference Images : ", number_of_reference_images)

    images_metercounter = []
    images_nonmechanical = []
    labels_metercounter = []
    labels_nonmechanical = []

    image_counter = 0
    for i in range(number_of_reference_images):
        # Definition for the first image and the next one
        current_first_image = ref_images[i]
        current_first_label = ref_labels[i]

        # Next number
        if i != number_of_reference_images - 1:
            next_first_image = ref_images[i + 1]
            next_first_label = ref_labels[i + 1]
        else:
            next_first_image = ref_images[0]
            next_first_label = ref_labels[0]

        for j in range(number_of_reference_images):

            # Definition for the second image and the next one
            current_second_image = ref_images[j]
            current_second_label = ref_labels[j]
            if j != number_of_reference_images - 1:
                next_second_image = ref_images[j + 1]
                next_second_label = ref_labels[j + 1]
            else:
                next_second_image = ref_images[0]
                next_second_label = ref_labels[0]

            # Build Initial Image
            auxiliary_image = concatenate_images(current_first_image, current_second_image)
            auxiliary_label = concatenate_classes(current_first_label, current_second_label)
            images_metercounter.append(auxiliary_image)
            labels_metercounter.append(auxiliary_label)
            images_nonmechanical.append(auxiliary_image)
            labels_nonmechanical.append(auxiliary_label)
            image_counter += 1

            #plotutils.plot_single_image(current_first_image, label="", shape=(310, 165))
            #plotutils.plot_single_image(current_second_image, label="", shape=(310, 165))
            #plotutils.plot_single_image(auxiliary_image, label="", shape=(310, 330))

            # Parameters
            parameters = np.arange(parameter_range[0], parameter_range[1], step)
            for parameter in parameters:
                second_label = np.where(current_second_label == 1)[1][0]
                para_second_image, para_second_label = generate_shifted_images(current_second_image, next_second_image,
                                                                               current_second_label, next_second_label,
                                                                               parameter, threshold)

                para_first_image = current_first_image
                para_first_label = current_first_label
                if second_label == 9:
                    para_first_image, para_first_label = generate_shifted_images(current_first_image, next_first_image,
                                                                                 current_first_label, next_first_label,
                                                                                 parameter, threshold)

                # Build Initial Image
                images_metercounter.append(concatenate_images(para_first_image, para_second_image))
                labels_metercounter.append(concatenate_classes(para_first_label, para_second_label))
                image_counter += 1
                parameter += step

    print("Number of generated images : ", image_counter)
    return np.array(images_metercounter), np.array(labels_metercounter), np.array(images_nonmechanical), np.array(labels_nonmechanical)


def generate_countermeter_dataset(ref_images, ref_labels, parameter_range, step, threshold):
    """
    Function used for generating a full dataset used in second experiment
    Generates a complete set of a parametrised meter counter from 00 to 99...
    :param ref_images:
    :param ref_labels:
    :return:
    """

    # Validation of the Parameters
    if not assert_parameters(parameter_range):
        print("Parameters are not valid!")
        return

    # print("Parameters are valid! Building dataset... Please wait!")
    # print("Reference Images Shape : ", ref_images.shape)
    # print("Reference Labels Shape : ", ref_labels.shape)
    number_of_reference_images = ref_images.shape[0]

    # print("Number of Reference Images : ", number_of_reference_images)

    images = []
    labels = []

    nonmechanical_images = []
    nonmechanical_labels = []

    counter = 0
    for i in range(number_of_reference_images):
        # Definition for the first image and the next one
        current_first_image = ref_images[i]
        current_first_label = ref_labels[i]

        # Next number
        if i != number_of_reference_images - 1:
            next_first_image = ref_images[i + 1]
            next_first_label = ref_labels[i + 1]
        else:
            next_first_image = ref_images[0]
            next_first_label = ref_labels[0]

        for j in range(number_of_reference_images):

            # Definition for the second image and the next one
            current_second_image = ref_images[j]
            current_second_label = ref_labels[j]
            if j != number_of_reference_images - 1:
                next_second_image = ref_images[j + 1]
                next_second_label = ref_labels[j + 1]
            else:
                next_second_image = ref_images[0]
                next_second_label = ref_labels[0]

            # Build Initial Image
            auxiliary_image = concatenate_images(current_first_image, current_second_image)
            auxiliary_label = concatenate_classes(current_first_label, current_second_label)
            images.append(auxiliary_image)
            labels.append(auxiliary_label)
            nonmechanical_images.append(auxiliary_image)
            nonmechanical_labels.append(auxiliary_label)
            counter += 1

            #plotutils.plot_single_image(current_first_image, label="", shape=(310, 165))
            #plotutils.plot_single_image(current_second_image, label="", shape=(310, 165))
            #plotutils.plot_single_image(auxiliary_image, label="", shape=(310, 330))

            # Parameters
            parameters = np.arange(parameter_range[0], parameter_range[1], step)
            for parameter in parameters:
                second_label = np.where(current_second_label == 1)[1][0]
                para_second_image, para_second_label = generate_shifted_images(current_second_image, next_second_image,
                                                                               current_second_label, next_second_label,
                                                                               parameter, threshold)

                para_first_image = current_first_image
                para_first_label = current_first_label
                if second_label == 9:
                    para_first_image, para_first_label = generate_shifted_images(current_first_image, next_first_image,
                                                                                 current_first_label, next_first_label,
                                                                                 parameter, threshold)

                # Build Initial Image
                images.append(concatenate_images(para_first_image, para_second_image))
                labels.append(concatenate_classes(para_first_label, para_second_label))
                counter += 1
                parameter += step

    print("Number of generated images : ", counter)
    return np.array(images), np.array(labels), np.array(nonmechanical_images), np.array(nonmechanical_labels)