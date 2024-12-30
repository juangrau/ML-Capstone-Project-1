import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

'''
#################################################

In order to run this script, it is important that 
you make sure that the image dataset is already on 
the same folder.

For this, you can execute from the command line the
following commands:

wget https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip

unzip -q EuroSAT_RGB.zip

This will download the dataset zip file and unzip it

#################################################
'''

categories = os.listdir('EuroSAT_RGB')
categories

for dir_name in ['train', 'val', 'test']:
    for cat in categories:
      os.makedirs(dir_name, exist_ok=True)
      os.makedirs(os.path.join(dir_name, cat), exist_ok=True)

for cat in categories:
  image_paths = []
  for img in os.listdir(os.path.join('EuroSAT_RGB/', cat)):
    image_paths.append(os.path.join('EuroSAT_RGB/', cat, img))
  print(cat, len(image_paths))
  full_train_paths, test_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
  train_paths, val_paths = train_test_split(full_train_paths, test_size=0.25, random_state=42)
  for path in train_paths:
    image = path.split('/')[-1]
    shutil.copy(path, os.path.join('train', cat, image))
  for path in val_paths:
    image = path.split('/')[-1]
    shutil.copy(path, os.path.join('val', cat, image))
  for path in test_paths:
    image = path.split('/')[-1]
    shutil.copy(path, os.path.join('test', cat, image))



# Define paths and parameters
main_dir = os.getcwd()
train_dir = os.path.join(main_dir, 'train')
test_dir = os.path.join(main_dir, 'test')
val_dir = os.path.join(main_dir, 'val')
img_width, img_height = 150, 150
batch_size = 32
num_classes = 10  # Adjust based on the number of classes
epochs = 35

# Create ImageDataGenerator objects
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.02,
    rotation_range=15,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

def make_model(learning_rate, dropout_rate=0.5):
    optimizer = Adam(learning_rate=learning_rate)
    model = Sequential()

    # First convolutional block
    model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second convolutional block
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third convolutional block (optional, for deeper models)
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

checkpoint = keras.callbacks.ModelCheckpoint(
    'model_vf_{epoch:02d}_{val_accuracy:.3f}.h5.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

learning_rate = 0.001
dropout_rate = 0.1
batch_size = 32
scores = {}

model = make_model(learning_rate=learning_rate, dropout_rate=dropout_rate)
history = model.fit(
  train_generator,
  steps_per_epoch=train_generator.samples // batch_size,
  epochs=epochs,
  validation_data=validation_generator,
  validation_steps=validation_generator.samples // batch_size,
  callbacks=[checkpoint])
scores['final'] = history.history
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print('Test accuracy:', test_acc)
