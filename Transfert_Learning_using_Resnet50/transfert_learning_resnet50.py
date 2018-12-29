import numpy as np
import os
import time
import keras 
from resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from imagenet_utils import preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
print(keras.__version__)

# ------------------------------------Loading the training data---------------------------#
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
		img_data_list.append(x)

img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)

#-----------------------------------Add label of image datasets classes and split them inot parts for training and test--------#
# Define the number of classes
num_classes = 4
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:202]=0
labels[202:404]=1
labels[404:606]=2
labels[606:]=3

names = ['cats','dogs','horses','humans']
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#------------------------------------------------Define the resnet custom model 1 : where we train ofnly the last layer -------------------#
# Custom_resnet_model_1
#Training the classifier alone
image_input = Input(shape=(224, 224, 3))

#general resnet amodel architecture 
model = ResNet50(input_tensor=image_input, require_flatten=True,weights='imagenet') # if you make require_flatten=False , you will remve the 2 laster layer (flastten and dense)
model.summary()

# custom resnet model 1
last_layer = model.get_layer('avg_pool').output
x= Flatten(name='flatten')(last_layer)
#define the output layer as function of the number of classes 
out = Dense(num_classes, activation='softmax', name='output_layer')(x)
custom_resnet_model = Model(inputs=image_input,outputs= out)
print('------------summary of the custom model 1---------------')
custom_resnet_model.summary()

# Exept the last layer we will freze the al previews layers 
for layer in custom_resnet_model.layers[:-1]:
	layer.trainable = False

custom_resnet_model.layers[-1].trainable
print('---summary of the freze model layer-------')
#custom_resnet_model.summary()

#--------------------Compile and trin the custom resnet model1--------------------#
'''# compile 
custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# train 
t=time.time()
hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
# evaluated the trained model adn get the scores 
(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
'''

#-----------------Method 2 for transfert learning based on resnet50: Fine tune the resnet 50----------------#
#image_input = Input(shape=(224, 224, 3))
model = ResNet50(weights='imagenet',require_flatten=False)
model.summary()
last_layer = model.output
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers
x = Dense(512, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)
# a softmax layer for 4 classes
out = Dense(num_classes, activation='softmax',name='output_layer')(x)

# this is the model we will train
custom_resnet_model2 = Model(inputs=model.input, outputs=out)

custom_resnet_model2.summary()

# Exept the 6 last layer we will freze the al previews layers 
for layer in custom_resnet_model2.layers[:-6]:
	layer.trainable = False

custom_resnet_model2.layers[-1].trainable

#--------------------Compile and trin the custom resnet model1--------------------#
custom_resnet_model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#train 
t=time.time()
hist = custom_resnet_model2.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
# evaluated the trained model adn get the scores 
(loss, accuracy) = custom_resnet_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

#save the model to h5
custom_resnet_model2.save_weights("custom_resnet50_model2.h5")
print("Saved model .h5 to disk")

# save the mode to json file
model_json = custom_resnet_model2.to_json()
with open("custom_vgg_model2.json", "w") as json_file:
    json_file.write(model_json)
print("Saved model json to disk")

#------------Plotthe analysis loss vs val and train vs acc -------------#
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(10)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])
plt.savefig("Fig_TL_resnet50_train_loss_vs_val_loss.png")


plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])
plt.savefig("Fig_TL_resnet50_train_acc_vs_val_acc.png")
