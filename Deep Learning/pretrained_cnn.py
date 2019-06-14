import keras
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from keras.preprocessing import image

from keras.layers import Input

 
#Load the VGG model
vgg_model = vgg16.VGG16(weights='imagenet')
 
#Load the Inception_V3 model
inception_model = inception_v3.InceptionV3(weights='imagenet')
 
#Load the ResNet50 model
resnet_model = resnet50.ResNet50(weights='imagenet')
 
#Load the MobileNet model
mobilenet_model = mobilenet.MobileNet(weights='imagenet')

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
%matplotlib inline
 
filename = 'images/cat.jpg'
# load an image in PIL format
original = load_img(filename, target_size=(224, 224))
print('PIL image size',original.size)
plt.imshow(original)
plt.show()
 
# convert the PIL image to a numpy array
# IN PIL - image is in (width, height, channel)
# In Numpy - image is in (height, width, channel)
numpy_image = img_to_array(original)
plt.imshow(np.uint8(numpy_image))
plt.show()
print('numpy array size',numpy_image.shape)
 
# Convert the image / images into batch format
# expand_dims will add an extra dimension to the data at a particular axis
# We want the input matrix to the network to be of the form (batchsize, height, width, channels)
# Thus we add the extra dimension to the axis 0.
image_batch = np.expand_dims(numpy_image, axis=0)
print('image batch size', image_batch.shape)
plt.imshow(np.uint8(image_batch[0]))



##########  VGG Model ######
processed_image = vgg16.preprocess_input(image_batch.copy())
predictions_vgg = vgg_model.predict(processed_image)
label = decode_predictions(predictions_vgg)
print("\n VGG \n", label)

########### ResNet 50 #############
predictions_resnet = resnet_model.predict(processed_image)
label = decode_predictions(predictions_resnet)
print("\n ResNet50", label)

########### MobileNet #############
predictions_mobilenet = mobilenet_model.predict(processed_image)
label = decode_predictions(predictions_mobilenet)
print("\n Mobile Net \n", label)

########### Inceptionv3 #############

from keras.applications.inception_v3 import preprocess_input

img_path = 'images/cat.jpg'
# load an image in PIL format
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)
 
predictions_inceptionv3 = inception_model.predict(x)
label = decode_predictions(predictions_inceptionv3)
print("\n Inception v3 \n", label)
