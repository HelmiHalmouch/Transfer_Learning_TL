'''
Here is an implimentation of Transfert Learning (TL) using VGG16
for classification of custom image datasets 

GHNAMI Helmi 
28/12/2018

'''

#import reuired librery and packages 
import time
import sys, os 
import matplotlib.pyplot as plt 
import numpy as np 
import keras 
from keras.utils import np_utils 
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras_applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#local model 
from vgg16 import VGG16

print('The packges are imported !')


#------------------------------Preprocessing and split of input images dataset (by from keras.preprocessing import image and train_test_split)---------------------------------#
# load the custum dataset
# Loading the training data
PATH = os.getcwd()
# Define data path
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)
img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
#		x = x/255
		#print('Input image shape:', x.shape)
		img_data_list.append(x)

img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)

# Define the number of classes
num_classes = 4
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:202]=0
labels[202:404]=1
labels[404:606]=2
labels[606:]=3

names = ['cats','dogs','horses','humans']

# convert class labels to on-hot encoding like [1 0 0 0] for the first class [0 1 0 0] for the second classe 
Y = np_utils.to_categorical(labels, num_classes)
#print(Y)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset train , test  (0.2 = 20% of the train data)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#-----------------------------Custum the model VGG16 for our dataset and classes------------------#
# Custom_vgg_model_1
#Training the classifier alone
image_input = Input(shape=(224, 224, 3))

model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()

#We have only 4 classes for our custom dataset, so we need to replace the last layer of the VGG16 model (preduction Dense)
print('Creation of the custom vgg model as function of our dataset')

last_layer = model.get_layer('fc2').output
out = Dense(num_classes, activation ='softmax', name='output')(last_layer)
custom_vgg_model = Model(image_input, out)
custom_vgg_model.summary()

# Freeze the model like the method 3 in transfert learning : to check you need to see the number of non trainbel parameters change from zero to other value in model.summary
for layer in custom_vgg_model.layers[:-1]: # all layers except the last layer 
	layer.trainable = False

custom_vgg_model.layers[3].trainable
print(custom_vgg_model.layers[3].trainable)

print('--------------summary of the custom model after the freeze step------------')
custom_vgg_model.summary()


'''# -----------------------compile and train the custom model----------------------------# 
custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

t=time.time()
#	t = now()
hist = custom_vgg_model.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))  # if we have valdation set we can repalce (X_test, y_test) by (X_validation, y_validation)
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

#save the model to h5
custom_vgg_model.save_weights("custom_vgg_model1.h5")
print("Saved model to disk")

# save the mode to json file
model_json = custom_vgg_model.to_json()
with open("custom_vgg_model.json", "w") as json_file:
    json_file.write(custom_vgg_model)
'''

####################################################################################################################
#----------------------------------Method 2 Use the fine tunnig for a lerger dataset--------------------#
print('------------------------Method 2----------------')
#Training the feature extraction also

image_input = Input(shape=(224, 224, 3))

model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')

model.summary()

last_layer = model.get_layer('block5_pool').output
x= Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
out = Dense(num_classes, activation='softmax', name='output')(x)
custom_vgg_model2 = Model(image_input, out)
custom_vgg_model2.summary()

# freez the  layer excepet the 3 last layer (which are modified in the custom_vgg_model2)
for layer in custom_vgg_model2.layers[:-3]:
	layer.trainable=False

custom_vgg_model2.summary()

# --------compile and train the custom model-------------# 
custom_vgg_model2.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

t=time.time()
#	t = now()
hist = custom_vgg_model2.fit(X_train, y_train, batch_size=32, epochs=3, verbose=1, validation_data=(X_test, y_test))  # if we have valdation set we can repalce (X_test, y_test) by (X_validation, y_validation)
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_vgg_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

#save the model to h5
custom_vgg_model2.save_weights("custom_vgg_model2.h5")
print("Saved model to disk")

# save the mode to json file
model_json = custom_vgg_model2.to_json()
with open("custom_vgg_model2.json", "w") as json_file:
    json_file.write(model_json)

#%%
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(3)


plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])
plt.savefig("Training_result_custom_TL2.png")

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])
plt.savefig("Training_result2_custom_TL2.png")
plt.show()
