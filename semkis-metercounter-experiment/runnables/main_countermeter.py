from utils import configutils, countermeterutils, plotutils, analysisutils, networkutils
from networks.countermeter_cnn import CNN

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import imageio, random
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Configure Initial Directories
configutils.initDirectories()

# Retrieve Reference Filenames
ref_img_filename = configutils.configSectionMap("DATA")['ref_images']
ref_labels_filename = configutils.configSectionMap("DATA")['ref_labels']

# Retrieve Reference Filenames for countermeter images and labels
ref_countermeter_images_filename = configutils.configSectionMap("DATA")['ref_countermeter_images']
ref_countermeter_labels_filename = configutils.configSectionMap("DATA")['ref_countermeter_labels']
x_train_filename = configutils.configSectionMap("DATA")['x_train_filename']
y_train_filename = configutils.configSectionMap("DATA")['y_train_filename']
x_dev_filename = configutils.configSectionMap("DATA")['x_dev_filename']
y_dev_filename = configutils.configSectionMap("DATA")['y_dev_filename']
x_test_filename = configutils.configSectionMap("DATA")['x_test_filename']
y_test_filename = configutils.configSectionMap("DATA")['y_test_filename']
ref_nonmec_images_filename = configutils.configSectionMap("DATA")['ref_nonmec_images']
ref_nonmec_labels_filename = configutils.configSectionMap("DATA")['ref_nonmec_labels']

x_train = countermeterutils.load_set_from_file(x_train_filename)
y_train = countermeterutils.load_set_from_file(y_train_filename)
x_dev = countermeterutils.load_set_from_file(x_dev_filename)
y_dev = countermeterutils.load_set_from_file(y_dev_filename)
x_test = countermeterutils.load_set_from_file(x_test_filename)
y_test = countermeterutils.load_set_from_file(y_test_filename)

x_train_nonmec_filename = configutils.configSectionMap("DATA")['x_train_nonmec_filename']
y_train_nonmec_filename = configutils.configSectionMap("DATA")['y_train_nonmec_filename']
x_dev_nonmec_filename = configutils.configSectionMap("DATA")['x_dev_nonmec_filename']
y_dev_nonmec_filename = configutils.configSectionMap("DATA")['y_dev_nonmec_filename']

# Creation and Saving of Reference Images
def create_and_save_refs():
    # Retrieve Reference Images & Format to Numpy Arrays
    path = configutils.configSectionMap("FOLDERS")['reference_dataset_path']
    ref_images, ref_labels = countermeterutils.collect_classified_data(path)

    # Print out a test version for a single digit
    index = 0
    plotutils.plot_single_image(ref_images[index], label=np.where(ref_labels[index] == 1)[1][0], shape=(310, 165))

    #save reference images as files
    countermeterutils.save_set_to_file(ref_img_filename, ref_images)
    countermeterutils.save_set_to_file(ref_labels_filename, ref_labels)
    return ref_images, ref_labels


def load_refs():
    ref_images = countermeterutils.load_set_from_file(ref_img_filename)
    ref_labels = countermeterutils.load_set_from_file(ref_labels_filename)
    index = 0
    plotutils.plot_single_image(ref_images[index], label=np.where(ref_labels[index] == 1)[1][0], shape=(310, 165))

    return ref_images, ref_labels


def save_and_build_countermeter_dataset(ref_images, ref_labels):
    # Load of Reference Images
    # Code for saving mechanical images to file
    # Sets of parameters
    parameter_range = (0.2, 0.8)
    step = 0.2
    threshold = 0.5

    # STEP 2 : Generate all mechanical images
    countermeter_images, countermeter_labels, nm_images, nm_labels = countermeterutils.generate_countermeter_dataset(ref_images, ref_labels, parameter_range, step, threshold)
    countermeterutils.save_set_to_file(ref_countermeter_images_filename, countermeter_images)
    countermeterutils.save_set_to_file(ref_countermeter_labels_filename, countermeter_labels)
    countermeterutils.save_set_to_file(ref_nonmec_images_filename, nm_images)
    countermeterutils.save_set_to_file(ref_nonmec_labels_filename, nm_labels)

    return countermeter_images, countermeter_labels, nm_images, nm_labels


def load_refs_metercounter():
    countermeter_images = countermeterutils.load_set_from_file(ref_countermeter_images_filename)
    countermeter_labels = countermeterutils.load_set_from_file(ref_countermeter_labels_filename)
    cm_ref_images = countermeterutils.load_set_from_file(ref_nonmec_images_filename)
    cm_ref_labels = countermeterutils.load_set_from_file(ref_nonmec_labels_filename)
    return countermeter_images, countermeter_labels, cm_ref_images, cm_ref_labels




"""indices = np.random.randint(101, size=5)

for index in indices:
    
    # aux = np.where(mechanical_labels[index] == 1)
    # label = aux[0][0]*10+aux[0][1]-10
    # print(label)
    # plotutils.plot_single_image(mechanical_images[index], label=label, shape=(310, 330))
    
    aux = np.where(y_train_nonmec[index] == 1)
    label = aux[0][0]*10+aux[0][1]-10
    print(label)
    plotutils.plot_single_image(x_train_nonmec[index], label=label, shape=(310, 330))

    aux = np.where(y_train_mec[index] == 1)
    label = aux[0][0]*10+aux[0][1]-10
    plotutils.plot_single_image(x_train_mec[index], label=label, shape=(310, 330))
"""


# Neural Network Trainining
def  targetNN_trainings(countermeter_cnn, nn_prefix, t_prefix, epochs = 50):
    # CNN Engineering and Training
    if nn_prefix is not None:
        countermeter_cnn.load_stored_cnn(cnn_name="cnn_trained_"+nn_prefix)
    else:
        countermeter_cnn.load_stored_cnn(cnn_name="untrained_cnn")

    # load_r1_datasets(countermeter_cnn)

    countermeter_cnn.NUM_EPOCH = epochs
    countermeter_cnn.train(filename="cnn_trained_"+t_prefix)
    countermeter_cnn.evaluate_model(filename=t_prefix+'_performance.txt')
    countermeter_cnn.generate_confusion_matrices(num_classes=100, filename=t_prefix)

    """
    # 2nd Round training iteration with mecanical images
    load_r2_datasets()
    countermeter_cnn.load_stored_cnn(cnn_name="cnn_trained_r1")

    countermeter_cnn.NUM_EPOCH = 25
    countermeter_cnn.train(filename="cnn_trained_r2")
    countermeter_cnn.evaluate_model(filename='r2_performance.txt')
    countermeter_cnn.generate_confusion_matrices(num_classes=100, filename="r2")
    """


def targetNN_analysis(countermeter_cnn, nn_prefix, round):
    countermeter_cnn.load_stored_cnn(cnn_name="cnn_trained_"+nn_prefix)
    #load_r1_datasets(countermeter_cnn)

    y_pred, y_pred_classes = networkutils.predict(countermeter_cnn.cnn, countermeter_cnn.x_test)
    analysisutils.generate_web_analystics(countermeter_cnn.x_test, countermeter_cnn.y_test, y_pred, round=round)
    """
    countermeter_cnn.load_stored_cnn(cnn_name="cnn_trained_r2")
    load_r2_datasets()
    y_pred, y_pred_classes = networkutils.predict(countermeter_cnn.cnn, countermeter_cnn.x_test)
    analysisutils.generate_web_analystics(countermeter_cnn.x_test, countermeter_cnn.y_test, y_pred, round=2)
    """

def targetNN_cm_generation(countermeter_cnn, nn_prefix):
    countermeter_cnn.load_stored_cnn(cnn_name="cnn_trained_"+nn_prefix)
    # load_r1_datasets(countermeter_cnn)
    countermeter_cnn.generate_confusion_matrices(num_classes=100, filename=nn_prefix)

    """
    countermeter_cnn.load_stored_cnn(cnn_name="cnn_trained_r2")
    load_r2_datasets()
    countermeter_cnn.generate_confusion_matrices(num_classes=100, filename="r2")
    """


def load_r1_datasets(countermeter_cnn):
    # 2nd Round training iteration with mecanical images
    countermeter_cnn.x_train = x_train
    countermeter_cnn.y_train = y_train
    countermeter_cnn.x_dev = x_dev
    countermeter_cnn.y_dev = y_dev
    countermeter_cnn.x_test = x_test
    countermeter_cnn.y_test = y_test


def print_datasetdetails(countermeter_cnn, prefix=""):
    analysisutils.plot_web_all_data(countermeter_cnn.x_train, countermeter_cnn.y_train, "training_"+prefix)
    analysisutils.plot_web_all_data(countermeter_cnn.x_dev, countermeter_cnn.y_dev, "developing_"+prefix)
    analysisutils.plot_web_all_data(countermeter_cnn.x_test, countermeter_cnn.y_test, "testing_"+prefix)

    print("x_train Shape : ", countermeter_cnn.x_train.shape)
    print("y_train Shape : ", countermeter_cnn.y_train.shape)
    print("x_dev Shape : ", countermeter_cnn.x_dev.shape)
    print("y_dev Shape : ", countermeter_cnn.y_dev.shape)
    print("x_test Shape : ", countermeter_cnn.x_test.shape)
    print("y_test Shape : ", countermeter_cnn.y_test.shape)


#Run the main file here :
if __name__ == '__main__':
    # ref_images, ref_labels = load_refs()
    # countermeter_images, countermeter_labels, nm_images, nm_labels = save_and_build_countermeter_dataset(ref_images, ref_labels)
    # countermeter_images, countermeter_labels, cm_ref_images, cm_ref_labels = load_refs_metercounter()
    # countermeter_images, countermeter_labels = countermeterutils.shuffle_set(countermeter_images, countermeter_labels)


    #analysisutils.plot_web_all_data(np.array(syn_images), np.array(syn_labels), filename="syn_images_r2")

    """
    #print(countermeter_images.shape)
    indices = np.random.randint(low=0, high=countermeter_images.shape[0], size=int(countermeter_images.shape[0]*0.3))

    print(len(indices))

    test_imgs = countermeter_images[indices]
    test_labels = countermeter_labels[indices]

    noisy_images, noisy_labels = analysisutils.generateSetOfNoisyImages(test_imgs, test_labels)

    countermeterutils.save_set_to_file("noisy_images", noisy_images)
    countermeterutils.save_set_to_file("noisy_labels", noisy_labels)

    countermeterutils.save_set_to_file("noisy_image_selection", test_imgs)
    countermeterutils.save_set_to_file("noisy_labels_selection", test_labels)

    final_images = np.concatenate((test_imgs, noisy_images))
    final_labels = np.concatenate((test_labels, noisy_labels))

   
    analysisutils.plot_web_all_data(countermeter_images, countermeter_labels, filename="cmall")
    imageio.mimsave('../results/analysis/cnn/data/countermeter-movie.gif', countermeter_images)

    analysisutils.plot_web_all_data(cm_ref_images, cm_ref_labels, filename="ncmall")
    imageio.mimsave('../results/analysis/cnn/data/ncmall-movie.gif', cm_ref_images)

    countermeter_images, countermeter_labels = countermeterutils.shuffle_set(countermeter_images, countermeter_labels)
    cm_ref_images, cm_ref_labels = countermeterutils.shuffle_set(cm_ref_images, cm_ref_labels)

    # Build Test Dataset
    random_seed = 2
    x_train, x_test, y_train, y_test = train_test_split(countermeter_images, countermeter_labels, test_size=0.2, random_state=random_seed)

    # Build Development and Training Dataset
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.1, random_state=random_seed)

    x_ref_train, x_ref_dev, y_ref_train, y_ref_dev = train_test_split(cm_ref_images, cm_ref_labels, test_size=0.1, random_state=random_seed)

    countermeterutils.save_set_to_file(x_train_filename, x_train)
    countermeterutils.save_set_to_file(y_train_filename, y_train)
    countermeterutils.save_set_to_file(x_dev_filename, x_dev)
    countermeterutils.save_set_to_file(y_dev_filename, y_dev)
    countermeterutils.save_set_to_file(x_test_filename, x_test)
    countermeterutils.save_set_to_file(y_test_filename, y_test)

    countermeterutils.save_set_to_file(x_train_nonmec_filename, x_ref_train)
    countermeterutils.save_set_to_file(y_train_nonmec_filename, y_ref_train)
    countermeterutils.save_set_to_file(x_dev_nonmec_filename, x_ref_dev)
    countermeterutils.save_set_to_file(y_dev_nonmec_filename, y_ref_dev)
    
    
    x_ref_train = countermeterutils.load_set_from_file(x_train_nonmec_filename)
    y_ref_train = countermeterutils.load_set_from_file(y_train_nonmec_filename)
    x_ref_dev = countermeterutils.load_set_from_file(x_dev_nonmec_filename)
    y_ref_dev = countermeterutils.load_set_from_file(y_dev_nonmec_filename)

    x_train = countermeterutils.load_set_from_file(x_train_filename)
    y_train = countermeterutils.load_set_from_file(y_train_filename)
    
    # Round 2 dataset : Added synthetic images for improving two digit shit
    x_train = np.concatenate((x_train, syn_images))
    y_train = np.concatenate((y_train, syn_labels))
    x_train, y_train = countermeterutils.shuffle_set(x_train, y_train)
    countermeterutils.save_set_to_file("x_train_r2",x_train)
    countermeterutils.save_set_to_file("y_train_r2",y_train)
    """



    """
    
    # Round 3 : Training CNN with optimal images and noisy images
    x_noisy_train, x_noisy_test, y_noisy_train, y_noisy_test = train_test_split(noisy_images, noisy_labels, test_size=0.2, random_state=2)

    noisy_images = countermeterutils.load_set_from_file("noisy_images")
    noisy_labels = countermeterutils.load_set_from_file("noisy_labels")
    
   
    x_test = np.concatenate((x_test, x_noisy_test))
    y_test = np.concatenate((y_test, y_noisy_test))
    countermeterutils.save_set_to_file("x_train_r3", x_train)
    countermeterutils.save_set_to_file("y_train_r3", y_train)
    countermeterutils.save_set_to_file("x_test_r3", x_test)
    countermeterutils.save_set_to_file("y_test_r3", y_test)
    # analysisutils.plot_web_all_data(np.array(noisy_images), np.array(noisy_labels), filename="test_noisy")
    

"""
    # Round 4 : Training CNN with optimal images, noisy images and synthetic images
    x_train = countermeterutils.load_set_from_file("x_train_r3")
    y_train = countermeterutils.load_set_from_file("y_train_r3")
    x_dev = countermeterutils.load_set_from_file(x_dev_filename)
    y_dev = countermeterutils.load_set_from_file(y_dev_filename)
    x_test = countermeterutils.load_set_from_file("x_test_r3")
    y_test = countermeterutils.load_set_from_file("y_test_r3")
    syn_images = countermeterutils.load_set_from_file("syn_images_r2")
    syn_labels = countermeterutils.load_set_from_file("syn_labels_r2")

    x_train = np.concatenate((x_train, syn_images))
    y_train = np.concatenate((y_train, syn_labels))

    countermeterutils.save_set_to_file("x_train_r4", x_train)
    countermeterutils.save_set_to_file("y_train_r4", y_train)
    countermeterutils.save_set_to_file("x_test_r4", x_test)
    countermeterutils.save_set_to_file("y_test_r4", y_test)

    countermeter_cnn = CNN(x_train, x_dev, x_test, y_train, y_dev, y_test)
    # analysisutils.plot_web_all_data(x_ref_train, y_ref_train, filename="ref_train")

    print_datasetdetails(countermeter_cnn, prefix="r4")
    targetNN_trainings(countermeter_cnn, epochs=25, nn_prefix="r3", t_prefix="r4")
    targetNN_analysis(countermeter_cnn, nn_prefix="r4", round=4)
    targetNN_cm_generation(countermeter_cnn, nn_prefix="r4")
