from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from ImageUtils.nn.conv.LeNet import LeNet
from ImageUtils.utils.captchahelper import preprocess

from imutils import paths
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import argparse
from datetime import datetime

ap= argparse.ArgumentParser()
ap.add_argument("-d","--dataset",help="Path to input dataset",required=True)
ap.add_argument("-m","--model",help="path to save the model",required=True)
args=vars(ap.parse_args())

data,labels=[],[]
print("[INFO..] Preprocessing the images ")
for imagePath in paths.list_images(args["dataset"]):
    image=cv2.imread(imagePath)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=preprocess(gray,28,28)
    image=img_to_array(image)
    data.append(image)
    
    label=imagePath.split(os.path.sep)[-2] #taking class label 
    #ex dataset/8/00003.jpg then list[-2] will be 8 that is the label of that image
    labels.append(label)
    
data=(np.array(data,dtype="float"))/255.0
labels=np.array(labels)

(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.25,random_state=42)

#applying one hot encoding
lb=LabelBinarizer().fit(trainY)
testY=lb.transform(testY)
trainY=lb.transform(trainY)

print("[INFO....] Compiling the model")
model=LeNet.build(width=28,height=28,depth=1,classes=9)
optimizer=SGD(lr=0.01)
model.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=["accuracy"])
print("\033[90m  {} \033[00m [INFO..] training the model ".format(datetime.now()))

H=model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=32,epochs=15,verbose=1)
#evaluating the model
print("\033[90m  {} \033[00m [INFO..] Evaluating the model ".format(datetime.now()))

predictions=model.predict(testX,batch_size=32)
print("\033[93m  {} \033[00m [INFO..]  Classification Accuracy ; {}".format(datetime.now(),classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=lb.classes_)))


print("\033[90m  {} \033[00m [INFO..] Saving the model ".format(datetime.now()))
model.save(args["model"])
#visualizing
plt.style.use("ggplot")
plt.figure()
plt.title("Training Loss and Accuracy")
plt.xlabel("# Epochs")
plt.ylabel("Loss / Accuarcy")
plt.plot(np.arange(0,15),H.history["loss"],label="Training loss")
plt.plot(np.arange(0,15),H.history["val_loss"],label="Testing loss")
plt.plot(np.arange(0,15),H.history["acc"],label="Training accuracy")
plt.plot(np.arange(0,15),H.history["val_acc"],label="Testing accuracy")
plt.legend()
plt.show()















    