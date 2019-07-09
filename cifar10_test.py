import os
from keras.datasets import cifar10
import keras.utils as utils
from keras.models import load_model
import numpy as np

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(_, _), (x_test, y_test) = cifar10.load_data()

x_test = x_test.astype('float32') / 255.0
y_test = utils.to_categorical(y_test)

cwd = os.getcwd()
model = load_model(filepath= cwd + "/image_classifier.h5")

# Optional: Show Accuracy and Loss
output = model.evaluate(x=x_test, y=y_test)
print("Test Loss:", output[0])
print("Test Accuracy:", output[1])

def img_predict(index):
    test_image = np.asarray([x_test[index]])
    predict_index = model.predict(x=test_image)
    index_maximum = np.argmax(predict_index[0])
    print("Image Prediction:", labels[index_maximum])

# Testing
# test_image = np.asarray([x_test[1]])
# predict_index = model.predict(x=test_image)
# print(predict_index[0])
# print(np.argmax(predict_index[0]))
