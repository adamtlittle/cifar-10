import os
from keras.datasets import cifar10
import keras.utils as utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD

# GPU Acceleration
from keras import backend as K
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
K.tensorflow_backend._get_available_gpus()

# Load Cifar Daya
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Format images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Make Values Categorical
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

# Creating a Sequential type model so we can add layers as we go
model = Sequential()

# Convolution Layers with relu activation
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same',
                 kernel_constraint=maxnorm(3)))

# MaxPooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten Layer
model.add(Flatten())

# Dense Layer - Higher Neurons (Units)  = More Accurate but Slower Training
model.add(Dense(units=2048, activation='relu', kernel_constraint=maxnorm(3)))

# Dropout Layer
model.add(Dropout(rate=0.5))

# Final Dense Layer / Output
# Number of Units  must equal number of possible Labels, in this case 10
# Softmax deals with probabilities for 10 categories.
model.add(Dense(units=10, activation='softmax'))

# Compiler with Stochastic Gradient Descent Optimizer
model.compile(optimizer=SGD(lr=.01), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model (30 - 50 epochs will train well)
model.fit(x=x_train, y=y_train, epochs=50, batch_size=32)

cwd = os.getcwd()
model.save(filepath=cwd + "/image_classifier.h5")


