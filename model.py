# -*- coding: utf-8 -*-

# tensorflow 2.0 beta and tensorflow datasets
!pip install tensorflow-gpu==2.0.0-beta1
!pip install tensorflow-datasets

# tensorflow
import tensorflow as     tf
from   tensorflow import keras

# tensorflow datasets
import tensorflow_datasets as tfds

# additional libraries
import math
import numpy             as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from keras.regularizers import l2

IMAGE_RES_mobilenet = 224
TRAINING_BATCH_SIZE = 32
IMAGE_ROWS = 224
IMAGE_COLS = 224
CHANNELS = 3

NUM_CLASSES = 1001
TRAINING_LR_MAX = 0.001

MODEL_LEVEL_0_BLOCKS    = 6
MODEL_LEVEL_1_BLOCKS    = 8
MODEL_LEVEL_2_BLOCKS    = 3

def create_model(level_0_repeats, level_1_repeats, level_2_repeats):
  
    # encoder - input
    model_input = keras.Input(shape=(IMAGE_ROWS, IMAGE_COLS, CHANNELS), name='input_image')
    x           = model_input
    
    # encoder - level 0
    for n0 in range(level_0_repeats):
        x = keras.layers.Conv2D(32, 3, strides=1, padding='same', activation=None, use_bias=False)(x)
        x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
        x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # encoder - level 1
    for n1 in range(level_1_repeats):
        x = keras.layers.Conv2D(64, 3, strides=1, padding='same', activation=None, use_bias=False)(x)
        x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
        x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # encoder - level 2
    for n2 in range(level_2_repeats):
        x = keras.layers.Conv2D(128, 3, strides=1, padding='same', activation=None, use_bias=False)(x)
        x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
        x = keras.layers.ReLU()(x)

    # encoder - output
    encoder_output = x

    # decoder
    y              = keras.layers.GlobalAveragePooling2D()(encoder_output)
    decoder_output = keras.layers.Dense(NUM_CLASSES, activation='softmax')(y)
    
    # forward path
    model = keras.Model(inputs=model_input, outputs=decoder_output, name='cifar_model')

    # loss, backward path (implicit) and weight update
    model.compile(optimizer=tf.keras.optimizers.Adam(TRAINING_LR_MAX), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def resizeData(img, label):
  img = tf.image.resize(
      img['image'],
      (224, 224),
      method=tf.compat.v2.image.ResizeMethod.BILINEAR
  )
  return img, label
  
# Getting a pretrained neural net
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", output_shape=[1001])
])
model.build([None, 224, 224, 3])  # Batch input shape.
model.summary()

# Downloading the downsampled dataset
dataset_train, info = tfds.load("downsampled_imagenet/64x64", split=tfds.Split.TRAIN, with_info=True)
dataset_test,  info = tfds.load("downsampled_imagenet/64x64", split='validation',  with_info=True)

# get labels for dataset_train
data_batches = dataset_train.batch(TRAINING_BATCH_SIZE).prefetch(2)

overall_label_arr = []
next_batch = iter(data_batches)

while True:
  try:
    train_batch = next(next_batch)
  except:
    break
  image_batch = train_batch['image'].numpy()

  image_batch = tf.image.resize( 
      image_batch,
      (224, 224),
      method=tf.compat.v2.image.ResizeMethod.BILINEAR
  )

  predicted_batch = model.predict(image_batch)
  predicted_batch = tf.squeeze(predicted_batch).numpy()
  predict_labels = np.argmax(predicted_batch, axis=1)
  overall_label_arr.extend(predict_labels)

d2 = tf.data.Dataset.from_tensor_slices(overall_label_arr)
train_dataset_labelled = tf.data.Dataset.zip((dataset_train, d2))

# get labels for dataset_test
test_batches = dataset_test.batch(TRAINING_BATCH_SIZE).prefetch(2)

overall_label_arr = []
next_batch = iter(test_batches)

while True:
  try:
    test_batch = next(next_batch)
  except:
    break
  image_batch = test_batch['image'].numpy()

  image_batch = tf.image.resize(
      image_batch,
      (224, 224),
      method=tf.compat.v2.image.ResizeMethod.BILINEAR
  )

  predicted_batch = model.predict(image_batch)
  predicted_batch = tf.squeeze(predicted_batch).numpy()
  predict_labels = np.argmax(predicted_batch, axis=1)
  overall_label_arr.extend(predict_labels)

d2 = tf.data.Dataset.from_tensor_slices(overall_label_arr)
test_dataset_labelled = tf.data.Dataset.zip((dataset_test, d2))
model = create_model(MODEL_LEVEL_0_BLOCKS, MODEL_LEVEL_1_BLOCKS, MODEL_LEVEL_2_BLOCKS)
model.summary()

# run the previous training set/image labels on the new model to train it

EPOCHS = 500
itr1 = train_dataset_labelled.batch(TRAINING_BATCH_SIZE).map(resizeData).prefetch(2)
print(train_dataset_labelled)
model.fit(itr1, steps_per_epoch=EPOCHS, verbose=1)

# now test using the previous validation set/image labels on the new model

EPOCHS = 5
itr1 = test_dataset_labelled.batch(TRAINING_BATCH_SIZE).map(resizeData).prefetch(2)
print(train_dataset_labelled)

test_data_loss, test_data_accuracy = model.evaluate(itr1)

print("Test data loss : " + str(test_data_loss) + " and Test data accuracy = " + str(test_data_accuracy))