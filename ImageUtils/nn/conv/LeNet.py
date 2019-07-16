##LeNet Architecu=ture is CONV==>TANH==>POOL==>CONV==>TANH==>POOL==>FC==>TANH==>FC==>SOFTMAX
from keras.models import Sequential
from keras.layers.core import Dense,Flatten,Activation
from keras import backend as K

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


class LeNet:
    @staticmethod
    def build(width,height,depth,classes):
        image=(width,height,depth)
        if K.image_data_format()=="channels_first":
            image=(depth,width,height)
        model = Sequential()
        model.add(Conv2D(20,(5,5),padding="same",input_shape=image))
        
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(50,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model