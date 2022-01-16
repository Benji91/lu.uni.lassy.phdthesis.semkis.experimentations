import numpy as np

from networks.cdcgan import CDCGAN
from networks.mnist_cnn import CNN
from utils import configutils, datautils


def main():
    configutils.initDirectories()

    # Only used once to build the datasets and to store it.
    # =====================================================
    x_train, y_train, x_test, y_test, x_dev, y_dev = datautils.mnist_dataset_construction()

    # Used to load the datasets and store them in the array.
    # ======================================================
    config_dataset_path = configutils.configSectionMap("FOLDERS")['dataset_path']
    config_data = configutils.configSectionMap("DATA")

    x_train = np.load(config_dataset_path + config_data['x_train'] + ".npy")
    y_train = np.load(config_dataset_path + config_data['y_train'] + ".npy")
    x_test = np.load(config_dataset_path + config_data['x_test'] + ".npy")
    y_test = np.load(config_dataset_path + config_data['y_test'] + ".npy")
    x_dev = np.load(config_dataset_path + config_data['x_dev'] + ".npy")
    y_dev = np.load(config_dataset_path + config_data['y_dev'] + ".npy")

    # Engineering of initial CNN and training it.
    # ===========================================
    # train_cnn(x_train, y_train, x_test, y_test, x_dev, y_dev)

    # Load TRAINED CNN and evaluate the model.
    # ========================================
    print(x_train.shape)
    print(y_train.shape)
    #build_trained_cnn(x_train, y_train, x_test, y_test, x_dev, y_dev)

    # Load UNTRAINED CNN and evaluate the model.
    # ==========================================
    # cnn_untrained = build_untrained_cnn(x_train, y_train, x_test, y_test, x_dev, y_dev)

    # Building a CDCGAN.
    # Note that the above section (loading an untrained CNN) needs to be uncommented.
    # build_cdcgan(cnn_untrained)
    # build_cdcgan()


def train_cnn(x_train, y_train, x_test, y_test, x_dev, y_dev):
    """ Engineering of initial CNN and training it. """
    cnn = CNN(x_train, y_train, x_test, y_test, x_dev, y_dev)
    cnn.build_cnn()

    cnn.train(epoch=1)
    cnn.generate_confusion_matrices()
    cnn.evaluate_model(filename="1st_trained_cnn_evaluation")


def build_trained_cnn(x_train, y_train, x_test, y_test, x_dev, y_dev):
    """ Load TRAINED CNN and evaluate the model. """
    cnn_trained = CNN(x_train, y_train, x_test, y_test, x_dev, y_dev)
    cnn_trained.build_trained_cnn()


def build_untrained_cnn(x_train, y_train, x_test, y_test, x_dev, y_dev):
    """ Load UNTRAINED CNN and evaluate the model. """
    cnn_untrained = CNN(x_train, y_train, x_test, y_test, x_dev, y_dev)
    cnn_untrained.build_untrained_cnn()

    # Only needed if we want to retrain the untrained model! Trained Model is saved!
    # cnn_untrained.train(filename="")
    # cnn_untrained.evaluate_model()
    # cnn_untrained.generate_confusion_matrices()

    return cnn_untrained


def build_cdcgan():
    """
    Building a CDCGAN.
    :return:
    """
    print("Start Training")
    cdcgan = CDCGAN()

    # epoch = 0
    # cdcgan.load_gan_for_epoch(epoch=epoch)
    # cdcgan.train(quantity=50, labels=[7], min_nb_pdiff=0, max_nb_pdiff=50, threshold=150, min_epoch=epoch)

    cdcgan.load_generator(epoch=340)

    number_of_images = 10000
    images, labels = cdcgan.generate_images(number_of_images=number_of_images, label=7)

    """
        # Filter images that produced erroneous data
        cnn_untrained.x_train = np.append(cnn_untrained.x_train, images, axis=0)
        cnn_untrained.y_train = np.append(cnn_untrained.y_train, labels, axis=0)
    
        cnn_untrained.train(filename=f"retrained_{number_of_images}_")
        cnn_untrained.evaluate_model(filename=f"retrained_{number_of_images}_")
        cnn_untrained.generate_confusion_matrices()
    """


if __name__ == "__main__":
    main()
