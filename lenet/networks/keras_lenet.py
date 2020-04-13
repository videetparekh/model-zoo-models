from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import BatchNormalization
from keras import backend as K
import keras



# create LeNet model using keras Sequential
def LeNet_keras(config):

    input = keras.layers.Input(shape=config.image_size)

    x = Conv2D(filters=6, kernel_size=5, padding='valid', use_bias=True, name='Conv1')(input)
    if config.do_batchnorm:
        x = BatchNormalization(name='bn_Conv1')(x)

    x = Activation('relu')(x)
    x = MaxPooling2D( strides=2, pool_size=2, name='max_pool1')(x)

    x = Conv2D(filters=16, kernel_size=5, padding='valid', use_bias=True, name='Conv2')(x)
    if config.do_batchnorm:
        x = BatchNormalization(name='bn_Conv2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(strides=2, pool_size=2, name='max_pool2')(x)

    x = Flatten()(x)

    x = Dense(units=120,use_bias=False, name='Dense1')(x)
    x = Activation(activation='relu', name='dense_relu1')(x)

    x = Dense(units=84, use_bias=False, name='Dense2')(x)
    x = Activation(activation='relu', name='dense_relu2')(x)

    x = Dense(units=config.num_classes, use_bias=True,
              name='Logits')(x)

    y = keras.layers.Softmax()(x)

    model = keras.Model(inputs=input, outputs=y)
    return model




# create LeNet model using keras Sequential
def mlp_keras(config):

    input = keras.layers.Input(shape=config.image_size)
    x = input

    #x = Conv2D(filters=6, kernel_size=5, padding='valid', use_bias=True, name='conv1')(x)
    #x = BatchNormalization(name='bn1', epsilon=1e-3)(x)
    #x = Activation('relu')(x)
    #x = MaxPooling2D( strides=2, pool_size=2, name='max_pool1')(x)

    x = Flatten()(x)

    #x = Dense(units=120,use_bias=False, name='dense1')(x)
    #x = Activation(activation='relu', name='dense_relu1')(x)

    x = Dense(units=config.num_classes, use_bias=True,
              name='Logits')(x)

    y = keras.layers.Softmax()(x)

    model = keras.Model(inputs=input, outputs=y)
    return model

