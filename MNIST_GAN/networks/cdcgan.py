from keras import models, layers
from keras.datasets import mnist
from keras.optimizers import Adam
import keras.utils as kerasutils
import numpy as np
from utils import configutils, cdcgan_utils, plotutils, datautils, networkutils
import logging


class DCGAN():

    def __init__(self):
        # Hyper-Parameters
        self.BATCH_SIZE = 128
        self.NUM_EPOCH = 5000
        self.LEARNING_RATE = 0.0002  # initial learning rate
        self.MOMENTUM = 0.5  # momentum term
        self.number_of_conditions = 10

        # Create Architecture for GAN componants
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.cdcgan = self.build_cdcgan(self.generator, self.discriminator)

        # Compiler Configuraiton
        optimizer = Adam(lr=self.LEARNING_RATE, beta_1=self.MOMENTUM)
        self.generator.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
        self.cdcgan.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)

        # Dataset Construction
        self.x_train, self.y_train, self.y = self.build_datasets()

    def build_stored_gan(self, trained: bool):
        """
        Building a Stored CDCGAN
        :return:
        """
        # Create Architecture for GAN componants
        if trained:
            self.generator = networkutils.load_model(configutils.configSectionMap("FOLDERS")['dcgan_generator_path'] +
                                                     configutils.configSectionMap("DCGAN")['dcgan_trained_gen'])
            self.discriminator = networkutils.load_model(
                configutils.configSectionMap("FOLDERS")['dcgan_discriminator_path'] +
                configutils.configSectionMap("DCGAN")['dcgan_trained_dis'])
        else:
            self.generator = networkutils.load_model(configutils.configSectionMap("FOLDERS")['dcgan_generator_path'] +
                                                     configutils.configSectionMap("DCGAN")['dcgan_untrained_gen'])
            self.discriminator = networkutils.load_model(
                configutils.configSectionMap("FOLDERS")['dcgan_discriminator_path'] +
                configutils.configSectionMap("DCGAN")['dcgan_untrained_dis'])

        # Discriminator Construction
        optimizer = Adam(lr=self.LEARNING_RATE, beta_1=self.MOMENTUM)
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)

        # DCGAN Construction
        self.discriminator.trainable = False
        dcgan = models.Sequential([self.generator, self.discriminator])
        dcgan.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)

    def build_cdcgan(self, generator, discriminator):
        """
        Build CDCGAN from skretch
        :param generator:
        :param discriminator:
        :return:
        """
        input_noise = layers.Input((100,))
        input_conditions = layers.Input((self.number_of_conditions,))

        gen_image = generator([input_noise, input_conditions])
        discriminator.trainable = False

        is_real = discriminator([gen_image, input_conditions])

        model = models.Model(inputs=[input_noise, input_conditions], outputs=is_real)

        # save models
        networkutils.save_model(self.generator,
                                configutils.configSectionMap("FOLDERS")['dcgan_generator_path'] +
                                configutils.configSectionMap("DCGAN")['dcgan_untrained_gen'])
        networkutils.save_model(self.discriminator,
                                configutils.configSectionMap("FOLDERS")['dcgan_discriminator_path'] +
                                configutils.configSectionMap("DCGAN")['dcgan_untrained_dis'])
        return model

    # Architecture of the Generator
    def build_generator(self, input_dim=100, units=1024, activation='relu'):
        """
        Construct the architecture of the Generator
        :param input_dim:
        :param units:
        :param activation:
        :return:
        """
        # Definition of the Generator for Input Noise
        lyr_noise_input = layers.Input(shape=(input_dim,))
        lyr_1_noise_Dense = layers.Dense(units)(lyr_noise_input)
        lyr_1_noise_Batch = layers.BatchNormalization()(lyr_1_noise_Dense)
        lyr_1_noise_act = layers.LeakyReLU(alpha=0.2)(lyr_1_noise_Batch)
        lyr_2_noise_Dense = layers.Dense(128 * 7 * 7)(lyr_1_noise_act)
        lyr_2_noise_Batch = layers.BatchNormalization()(lyr_2_noise_Dense)
        lyr_2_noise_reshape = layers.Reshape((7, 7, 128), input_shape=(128 * 7 * 7,))(lyr_2_noise_Batch)

        # Definition of the Generator for Input Conditions
        lyr__cond_input = layers.Input((self.number_of_conditions,))
        lyr_1_cond_dense = layers.Dense(1024)(lyr__cond_input)
        lyr_1_cond_act = layers.LeakyReLU(alpha=0.2)(lyr_1_cond_dense)
        lyr_2_cond_dense = layers.Dense(128 * 7 * 7)(lyr_1_cond_act)
        lyr_2_cond_batch = layers.BatchNormalization()(lyr_2_cond_dense)
        lyr_2_cond_reshape = layers.Reshape((7, 7, 128), input_shape=(128 * 7 * 7,))(lyr_2_cond_batch)

        # Combination of the two input sources
        lyr_concat = layers.Concatenate()([lyr_2_noise_reshape, lyr_2_cond_reshape])

        lyr_3_ups = layers.UpSampling2D((2, 2))(lyr_concat)
        lyr_3_con = layers.Conv2D(64, (5, 5), padding='same')(lyr_3_ups)
        lyr_3_act = layers.LeakyReLU(alpha=0.2)(lyr_3_con)
        lyr_4_ups = layers.UpSampling2D((2, 2))(lyr_3_act)
        lyr_4_con = layers.Conv2D(1, (5, 5), padding='same')(lyr_4_ups)
        lyr_4_act = layers.Activation('tanh')(lyr_4_con)

        # Complete networks
        model = models.Model(inputs=[lyr_noise_input, lyr__cond_input], outputs=lyr_4_act)
        logging.debug(model.summary())

        return model

    # Architecture of the Discriminator
    def build_discriminator(self, input_shape=(28, 28, 1), nb_filter=64, activation='relu'):
        """
        Construct the architecture of the discriminator
        :param input_shape:
        :param nb_filter:
        :param activation:
        :return:
        """
        # Definition of the layers for input images
        lyr_input_syn_image = layers.Input(shape=input_shape)
        lyr_1_conv = layers.Conv2D(nb_filter, (5, 5), padding='same')(lyr_input_syn_image)
        lyr_1_act = layers.LeakyReLU(alpha=0.2)(lyr_1_conv)
        lyr_1_pool = layers.MaxPooling2D(pool_size=(2, 2))(lyr_1_act)
        lyr_2_conv = layers.Conv2D(2 * nb_filter, (5, 5))(lyr_1_pool)
        lyr_2_act = layers.LeakyReLU(alpha=0.2)(lyr_2_conv)
        lyr_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(lyr_2_act)

        # Definition of the layers for conditions
        lyr_input_cond_input = layers.Input(shape=(self.number_of_conditions,))
        lyr_1_dense = layers.Dense(1024)(lyr_input_cond_input)
        lyr_1_act_con = layers.LeakyReLU(alpha=0.2)(lyr_1_dense)
        lyr_2_dense = layers.Dense(5 * 5 * 128)(lyr_1_act_con)
        lyr_2_reshape = layers.Reshape((5, 5, 128))(lyr_2_dense)

        # Concatenating the layers
        lyr_concat = layers.concatenate([lyr_3_pool, lyr_2_reshape])

        # Classifier definition
        lyr_3_flat = layers.Flatten()(lyr_concat)
        lyr_4_dense = layers.Dense(4 * nb_filter)(lyr_3_flat)
        lyr_5_elu = layers.LeakyReLU(alpha=0.2)(lyr_4_dense)
        lyr_6_dense = layers.Dense(1)(lyr_5_elu)
        lyr_6_act = layers.Activation('sigmoid')(lyr_6_dense)

        # Complete networks
        model = models.Model(inputs=[lyr_input_syn_image, lyr_input_cond_input], outputs=lyr_6_act)
        logging.debug(model.summary())
        return model

    def build_datasets(self):
        (x_train, y_train), (_, _) = mnist.load_data()

        # normalize images - All values from -1 to 1
        x_train = cdcgan_utils.normalise_images(x_train)
        x_train = x_train[:, :, :, None]

        # Converting y_train to a categorical expression
        y_train = kerasutils.to_categorical(y_train, self.number_of_conditions)

        # Initialising the output values
        y = [1] * self.BATCH_SIZE + [0] * self.BATCH_SIZE

        return x_train, y_train, y

    def predict(self, label):
        x_noise = cdcgan_utils.generate_noise((1, 100))
        x_category = np.zeros(10)
        x_category[label] = 1
        x_category = x_category.reshape((1, 10))
        syn_image = self.generator.predict(x=[x_noise, x_category], verbose=0)
        return syn_image

    def train(self, quantity, labels, min_nb_pdiff, max_nb_pdiff, threshold, min_epoch=0):
        """
        Train the CDCGAN
        :return:
        """
        nb_of_iterations_per_epoch = int(self.x_train.shape[0] / self.BATCH_SIZE)

        print("-------------------")
        print("Total epoch:", self.NUM_EPOCH, "Number of Iterations per Epoch:", nb_of_iterations_per_epoch)
        print("-------------------")

        iteration = 0

        # Array initialization for logging of the losses
        d_loss_logs = []
        g_loss_logs = []

        d_accuracy_logs = []
        g_accuracy_logs = []

        computed_accuracy = []

        for epoch in list(map(lambda x: x + 1, range(min_epoch, self.NUM_EPOCH))):
            # pbar = tqdm(desc="Epoch: {0}".format(epoch), total=X_train.shape[0])
            for i in range(nb_of_iterations_per_epoch):
                noise = cdcgan_utils.generate_noise((self.BATCH_SIZE, 100))
                image_batch = self.x_train[i * self.BATCH_SIZE:(i + 1) * self.BATCH_SIZE]
                label_batch = self.y_train[i * self.BATCH_SIZE:(i + 1) * self.BATCH_SIZE]

                synthetic_images = self.generator.predict([noise, label_batch], verbose=0)
                # SHAPES = (128,100) / (128,10)

                # Preparing data for discriminator training
                X = np.concatenate((image_batch, synthetic_images))
                label_batches_for_discriminator = np.concatenate((label_batch, label_batch))

                # train discriminator
                d_loss = self.discriminator.train_on_batch([X, label_batches_for_discriminator], self.y)

                # train generator
                noise = cdcgan_utils.generate_noise((self.BATCH_SIZE, 100))
                self.discriminator.trainable = False
                g_loss = self.cdcgan.train_on_batch([noise, label_batch], [1] * self.BATCH_SIZE)
                self.discriminator.trainable = True

                if i == 0:
                    cdcgan_utils.show_progress(epoch, i, g_loss[0], d_loss[0], g_loss[1], d_loss[1])
                    # pbar.update(self.BATCH_SIZE)
                """
                if i % 20 == 0:
                    d_loss_logs.append([iteration, d_loss[0]])
                    g_loss_logs.append([iteration, g_loss[0]])

                    d_accuracy_logs.append([iteration, d_loss[1]])
                    g_accuracy_logs.append([iteration, g_loss[1]])
                """
                iteration += 1

            d_loss_logs.append([iteration, d_loss[0]])
            g_loss_logs.append([iteration, g_loss[0]])

            d_accuracy_logs.append([iteration, d_loss[1]])
            g_accuracy_logs.append([iteration, g_loss[1]])

            cdcgan_utils.show_progress(epoch, i, g_loss[0], d_loss[0], g_loss[1], d_loss[1])
            csim, mse, ssim = datautils.quality_of_generator(generator=self.generator, quantity=quantity, labels=labels,
                                                             min_nb_pdiff=min_nb_pdiff, max_nb_pdiff=max_nb_pdiff,
                                                             threshold=threshold)
            computed_accuracy.append([iteration, csim, mse, ssim])
            print("Average Qualities of Generator CSIM : " + str(csim))
            print("Average Qualities of Generator MSE : " + str(mse))
            print("Average Qualities of Generator SSIM : " + str(ssim))

            # Save a generated image for every epoch
            if epoch == 1 or epoch % 10 == 0:
                cdcgan_utils.plot_set_of_images(self.generator, self.number_of_conditions,
                                                filename="image_epoch_" + str(epoch), show_figure=False)
                # save models
                networkutils.save_model(self.generator,
                                        configutils.configSectionMap("FOLDERS")['dcgan_generator_path'] +
                                        configutils.configSectionMap("DCGAN")['dcgan_trained_gen'] + str(epoch))
                networkutils.save_model(self.discriminator,
                                        configutils.configSectionMap("FOLDERS")['dcgan_discriminator_path'] +
                                        configutils.configSectionMap("DCGAN")['dcgan_trained_dis'] + str(epoch))
            if epoch % 10 == 0:
                plotutils.plot_training_statistics(d_loss_logs, g_loss_logs, d_accuracy_logs, g_accuracy_logs)
                plotutils.plot_generator_quality(computed_accuracy)

    def load_generator(self, epoch):
        self.generator = networkutils.load_model(configutils.configSectionMap("FOLDERS")['dcgan_generator_path'] +
                                                 configutils.configSectionMap("DCGAN")['dcgan_trained_gen'] + str(
            epoch))
        logging.debug(self.generator.summary())
        # self.generator.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

    def load_gan_for_epoch(self, epoch):
        self.load_generator(epoch=epoch)
        self.discriminator = networkutils.load_model(
            configutils.configSectionMap("FOLDERS")['dcgan_discriminator_path'] +
            configutils.configSectionMap("DCGAN")['dcgan_trained_dis'] + str(epoch))

        input_noise = layers.Input((100,))
        input_conditions = layers.Input((self.number_of_conditions,))

        gen_image = self.generator([input_noise, input_conditions])
        self.discriminator.trainable = False
        is_real = self.discriminator([gen_image, input_conditions])
        self.cdcgan = models.Model(inputs=[input_noise, input_conditions], outputs=is_real)

        # Compiler Configuraiton
        optimizer = Adam(lr=self.LEARNING_RATE, beta_1=self.MOMENTUM)
        self.generator.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
        self.cdcgan.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)

    def generate_images(self, number_of_images, label):
        images = []
        label_vectors = []
        for i in range(number_of_images):
            img = self.predict(label)
            images.extend(img)
            label_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            label_vector[label] = 1
            label_vectors.append(label_vector)

        np.save(configutils.configSectionMap("FOLDERS")['dataset_path'] + configutils.configSectionMap("DATA")['x_synthetic'] + "_" + str(number_of_images),
            np.array(images))
        np.save(configutils.configSectionMap("FOLDERS")['dataset_path'] + configutils.configSectionMap("DATA")['y_synthetic'] + "_" + str(number_of_images),
            np.array(label_vectors))

        return images, label_vectors
