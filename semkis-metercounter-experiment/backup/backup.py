"""

#BACKUP CODE
        # Retrieve filesnames for mechanical images
        mec_ref_img_filename = configutils.configSectionMap("DATA")['mec_ref_images']+"_p"+str(parameter)+"_th"+str(threshold)
        mec_ref_labels_filename = configutils.configSectionMap("DATA")['mec_ref_labels']+"_p"+str(parameter)+"_th"+str(threshold)

        # Save mechanical labels to file
        save_set_to_file(mec_ref_img_filename, mechanical_images)
        save_set_to_file(mec_ref_labels_filename, equivalence_classes)

  # Reshape to 3D Numpy Array
    images = images.reshape(number_of_references, height_ref_img, width_ref_img, 1)


    plt.imshow(images[0].reshape(310,165), cmap='gray')
    plt.title("test")
    plt.axis("off")
    plt.show()
"""

"""
for i in range(45, 65):
    label = str(np.where(mechanical_labels[i] == 1)[1][0])+str(np.where(mechanical_labels[i] == 1)[1][1]-10)
    print(label)
    plotutils.plot_single_image(mechanical_images[i], label=label, shape=(310, 330))

"""
"""


#STEP 1 : Build all ideal references images of a meter counter
metercounter_images, metercounter_labels = countermeterutils.generate_concatenated_images(ref_images, ref_labels)
plotutils.plot_single_image(metercounter_images[index], label=np.where(metercounter_labels[index] == 1)[1][0], shape=(310, 330))

#Save Reference Dataset Of Mechanical Images
countermeterutils.save_set_to_file(configutils.configSectionMap("DATA")['ref_mec_images'], metercounter_images)
countermeterutils.save_set_to_file(configutils.configSectionMap("DATA")['ref_mec_labels'], metercounter_labels)


#STEP 2 : Build all images that model a parametrised mechanical shift of a meter counter
parameter_range = (0.1, 0.9)
step = 0.1
threshold = 0.5
shifted_images, shifted_labels = countermeterutils.generate_rangeof_shifted_images(ref_images, ref_labels, parameter_range, step, threshold)

#Save plot to numpy array file

for i in range(0, 20):
    plotutils.plot_single_image(shifted_images[i], label=np.where(shifted_labels[i] == 1)[1][0], shape=(310, 165))

#STEP3 Generate all mechanical images
mechanical_images, mechanical_labels = countermeterutils.generate_countermeter_dataset(ref_images, ref_labels)

for i in range(0, 20):
    plotutils.plot_single_image(mechanical_images[i], label=np.where(mechanical_labels[i] == 1)[1][0], shape=(310, 165))

"""
"""
#STEP 3 : Build dataset with ideal & mechanical images of metercounter

brightness_range = (10, 250)
brightness_step = 10
countermeterutils.plotbrightness(ref_images[0].reshape(310, 165), str(ref_labels[0]), brightness_range, brightness_step)

countermeterutils.plotnoise(ref_images[0].reshape(310, 165), "localvar", str(ref_labels[0]))
countermeterutils.plotnoise(ref_images[0].reshape(310, 165), "poisson", str(ref_labels[0]))
countermeterutils.plotnoise(ref_images[0].reshape(310, 165), "salt", str(ref_labels[0]))
countermeterutils.plotnoise(ref_images[0].reshape(310, 165), "pepper", str(ref_labels[0]))
countermeterutils.plotnoise(ref_images[0].reshape(310, 165), "s&p", str(ref_labels[0]))
countermeterutils.plotnoise(ref_images[0].reshape(310, 165), "speckle", str(ref_labels[0]))
countermeterutils.plotnoise(ref_images[0].reshape(310, 165), None, str(ref_labels[0]))
"""
"""
index = 28
plt.title(metercounter_labels[index])
plt.imshow(metercounter_images[index].reshape(310,330), cmap='gray')
plt.show()


#Define x_train, y_train ; x_test, y_test ; x_dev, y_dev
x_train, x_dev, y_train, y_dev = train_test_split(metercounter_images, metercounter_labels, test_size=0.2, random_state=2)


#Instantiate the Convolutional Neural Network and run it
countermeter_cnn = CNN(x_train, x_dev, y_train, y_dev)
history = countermeter_cnn.train()
print(history.history['accuracy'])


"""


"""
index = 66

plotutils.plot_single_image(mechanical_images[index], label="test", shape=(310, 330))
plotutils.plot_single_image(mechanical_images[index+12], label="test", shape=(310, 330))
plotutils.plot_single_image(mechanical_images[index-19], label="test", shape=(310, 330))

analysisutils.generate_web_analystics(mechanical_images, mechanical_labels, mechanical_labels, round=0)


# Retrieve Reference Filenames
ref_img_filename = configutils.configSectionMap("DATA")['ref_images']
ref_labels_filename = configutils.configSectionMap("DATA")['ref_labels']

# Retrieve Reference Filenames of the images and labels
ref_mec_images_filename = configutils.configSectionMap("DATA")['ref_mec_images']
ref_mec_labels_filename = configutils.configSectionMap("DATA")['ref_mec_labels']
ref_nonmec_images_filename = configutils.configSectionMap("DATA")['ref_nonmec_images']
ref_nonmec_labels_filename = configutils.configSectionMap("DATA")['ref_nonmec_labels']

# Retrieve filenames for datasets
x_train_mec_filename = configutils.configSectionMap("DATA")['x_train_mec_filename']
y_train_mec_filename = configutils.configSectionMap("DATA")['y_train_mec_filename']
x_dev_mec_filename = configutils.configSectionMap("DATA")['x_dev_mec_filename']
y_dev_mec_filename = configutils.configSectionMap("DATA")['y_dev_mec_filename']
x_test_filename = configutils.configSectionMap("DATA")['x_test_filename']
y_test_filename = configutils.configSectionMap("DATA")['y_test_filename']
x_train_nonmec_filename = configutils.configSectionMap("DATA")['x_train_nonmec_filename']
y_train_nonmec_filename = configutils.configSectionMap("DATA")['y_train_nonmec_filename']
x_dev_nonmec_filename = configutils.configSectionMap("DATA")['x_dev_nonmec_filename']
y_dev_nonmec_filename = configutils.configSectionMap("DATA")['y_dev_nonmec_filename']


# Save Reference Dataset To File
countermeterutils.save_set_to_file(ref_img_filename, ref_images)
countermeterutils.save_set_to_file(ref_labels_filename, ref_labels)

"""


"""
# Code for saving mechanical images to file
# Sets of parameters
parameter_range = (0.2, 0.8)
step = 0.2
threshold = 0.5

# STEP 2 : Generate all mechanical images
mechanical_images, mechanical_labels, nonmechanical_images, nonmechanical_labels = countermeterutils.generate_countermeter_dataset(ref_images, ref_labels,
                                                                                       parameter_range, step, threshold)

# Save Reference Dataset To File
countermeterutils.save_set_to_file(ref_mec_images_filename, mechanical_images)
countermeterutils.save_set_to_file(ref_mec_labels_filename, mechanical_labels)
countermeterutils.save_set_to_file(ref_nonmec_images_filename, nonmechanical_images)
countermeterutils.save_set_to_file(ref_nonmec_labels_filename, nonmechanical_labels)

mechanical_images = countermeterutils.load_set_from_file(ref_mec_images_filename)
mechanical_labels = countermeterutils.load_set_from_file(ref_mec_labels_filename)
nonmechanical_images = countermeterutils.load_set_from_file(ref_nonmec_images_filename)
nonmechanical_labels = countermeterutils.load_set_from_file(ref_nonmec_labels_filename)

print("-------------------------")
print("Reference Images : ")
print(ref_images.shape)
print(ref_labels.shape)
print("-------------------------")
print("Mechanical Images : ")
print(mechanical_images.shape)
print(mechanical_labels.shape)
print("-------------------------")
print("Nonmechanical Images : ")
print(nonmechanical_images.shape)
print(nonmechanical_labels.shape)
print("-------------------------")


# Randomly Shuffled Mechanical ImagesImages
mechanical_images, mechanical_labels = countermeterutils.shuffle_set(mechanical_images, mechanical_labels)
nonmechanical_images, nonmechanical_labels = countermeterutils.shuffle_set(nonmechanical_images, nonmechanical_labels)

# Build fictive independent test dataset provided by customer
random_seed = 2
x_train_mec, x_test, y_train_mec, y_test = train_test_split(mechanical_images, mechanical_labels, test_size=0.2, random_state=random_seed)

# Build Mechanical and NonMechanical trainingdataset
x_train_mec, x_dev_mec, y_train_mec, y_dev_mec = train_test_split(x_train_mec, y_train_mec, test_size=0.1, random_state=random_seed)
x_train_nonmec, x_dev_nonmec, y_train_nonmec, y_dev_nonmec = train_test_split(nonmechanical_images, nonmechanical_labels, test_size=0.1, random_state=random_seed)


countermeterutils.save_set_to_file(x_train_mec_filename, x_train_mec)
countermeterutils.save_set_to_file(y_train_mec_filename, y_train_mec)
countermeterutils.save_set_to_file(x_dev_mec_filename, x_dev_mec)
countermeterutils.save_set_to_file(y_dev_mec_filename, y_dev_mec)

countermeterutils.save_set_to_file(x_test_filename, x_test)
countermeterutils.save_set_to_file(y_test_filename, y_test)
countermeterutils.save_set_to_file(x_train_nonmec_filename, x_train_nonmec)
countermeterutils.save_set_to_file(y_train_nonmec_filename, y_train_nonmec)
countermeterutils.save_set_to_file(x_dev_nonmec_filename, x_dev_nonmec)
countermeterutils.save_set_to_file(y_dev_nonmec_filename, y_dev_nonmec)
"""
"""
x_train_mec = countermeterutils.load_set_from_file(x_train_mec_filename)
y_train_mec = countermeterutils.load_set_from_file(y_train_mec_filename)
x_dev_mec = countermeterutils.load_set_from_file(x_dev_mec_filename)
y_dev_mec = countermeterutils.load_set_from_file(y_dev_mec_filename)
x_test = countermeterutils.load_set_from_file(x_test_filename)
y_test = countermeterutils.load_set_from_file(y_test_filename)
x_train_nonmec = countermeterutils.load_set_from_file(x_train_nonmec_filename)
y_train_nonmec = countermeterutils.load_set_from_file(y_train_nonmec_filename)
x_dev_nonmec = countermeterutils.load_set_from_file(x_dev_nonmec_filename)
y_dev_nonmec = countermeterutils.load_set_from_file(y_dev_nonmec_filename)
"""