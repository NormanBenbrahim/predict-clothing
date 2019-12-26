from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf 
import numpy as np
import cv2
import seaborn as sns 
import logging
import tensorflow_datasets as tfds
tfds.disable_progress_bar() # less busy this way
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# load the entire data
dataset, metadata = tfds.load('fashion_mnist', 
                              as_supervised=True, 
                              with_info=True)

# split into train (60K imgs) & test (10K imgs)
# each img is 28 x 28 pixels and labelled as a piece of clothing

# luckily the lovely folk at MNIST did this split for us:
train, test = dataset['train'], dataset['test']

# names of clothing
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 
               'Ankle boot']
class_pairs = {i:class_names[i] for i in range(10)}

# function to normalize images from [0,256] -> [0,1]
def normalize(imgs, labels):
    imgs = tf.cast(imgs, tf.float32) # convert to float
    imgs /= 255
    return imgs, labels

# apply the function on the datasets using the "map" method
train = train.map(normalize)
test = test.map(normalize)

# add the data to the cache so it doesn't have to re-load
train = train.cache()
test = test.cache()


# build the neural network layers
# we will build a convolutional neural network with 7 layers
all_layers = [] # initiate for appending

# don't worry what these mean for now - you can always
# tweak and read documents, and that's how you learn
conv_l1 = tf.keras.layers.Conv2D(32, 
                                 (3, 3), 
                                 padding='same',
                                 activation=tf.nn.relu,
                                 input_shape=(28, 28, 1))
all_layers.append(conv_l1)

maxpool_l1 = tf.keras.layers.MaxPooling2D((2, 2),
                                          strides=2)
all_layers.append(maxpool_l1)

conv_l2 = tf.keras.layers.Conv2D(64, 
                                 (3, 3), 
                                 padding='same', 
                                 activation=tf.nn.relu)
all_layers.append(conv_l2)

maxpool_l2 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)
all_layers.append(maxpool_l2)

flat_layer = tf.keras.layers.Flatten()
all_layers.append(flat_layer)

hidden = tf.keras.layers.Dense(128, activation=tf.nn.relu)
all_layers.append(hidden)

output = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
all_layers.append(output)

# initiate the model with all the layers
model = tf.keras.Sequential(all_layers)

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# repeat forever (not actually, see below)
train = train.repeat()

# shuffle the data so that the model doesn't think order plays a role
train_size = 60000
train = train.shuffle(train_size)

# define a batch (step) size, then tell the model to use that many
# batches to update model variables in both the test & train data
batch_size = 32
train = train.batch(batch_size)
test = test.batch(batch_size)

#### fit the model to the training data
# the epoch param will make sure 'train.repeat()' only lasts the 
# number of epochs and doesn't go on for infinity to find a solution
model.fit(train, 
          epochs=5,
          steps_per_epoch=np.ceil(train_size/batch_size))

# now apply the model to the test data 
results = model.evaluate(test,
                         steps=np.ceil(train_size/batch_size))
test_loss, test_accuracy = results

# print accuracy
print('\n')
print('Accuracy on test data: ', test_accuracy)

#### now try predictions on real images
# this doesn't work very well: the original images suck
# so it flops on real world data & thinks everything is a bag
def predict_new(img_file):
    img = cv2.imread(img_file, 0)
    img = img.astype('float32')
    img = img/256 # normalize

    # resize image to 28x28, 4 is code for lanczos interp
    img_resized = cv2.resize(img, (28,28), interpolation=4)

    # add extra dimensions needed for the model & make 
    # shape --> (1, 28, 28, 1)
    img_resized = np.expand_dims(img_resized, 2)
    img_resized = np.array([img_resized])
    
    # make prediction
    predict = model.predict(img_resized)
    index = predict.argmax()

    return "Predicted class '{}': {}".format(predict.argmax(),
                                             class_pairs[index])


predict_new('imgs/red_heels.jpeg')






