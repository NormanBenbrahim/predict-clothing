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

# function to normalize images from [0,256] -> [0,1]
def normalize(imgs, labels):
    imgs = tf.cast(imgs, tf.float32) # convert to float
    imgs /= 255
    return imgs, labels

# apply the function on the datasets using the "map" method
train = train.map(normalize)
test = test.map(normalize)

# build the neural network layers
input_layer = tf.keras.layers.Flatten(input_shape=(28, 28, 1))
hidden_layer = tf.keras.layers.Dense(128, activation=tf.nn.relu)
output_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

# initiate the model with the layers
model = tf.keras.Sequential([input_layer, hidden_layer, output_layer])

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
def predict_new(img_file):
    img = cv2.imread(img_file, 0)
    img = img.astype('float32')
    img = img/256

    # resize image to 28x28, 4 is code for lanczos interp
    img_resized = cv2.resize(img, (28,28), interpolation=4)

    # add extra dimensions needed for the model & make 
    # shape --> (1, 28, 28, 1)
    img_resized = np.expand_dims(img_resized, 2)
    img_resized = np.array([img_resized]) # shape -> (1, 28, 28, 1)
    
    # make prediction
    predict = model.predict(img_resized)

    return predict 






