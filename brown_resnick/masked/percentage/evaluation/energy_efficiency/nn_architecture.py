#This python script trains the neural network (CNN) on the classification task for a machine with one gpu (cuda device).
#See br_nn_with_distributed_training.py for training the neural network with multiple gpus.
import tensorflow
from tensorflow import keras
from keras import layers, models, Input
import os
import json


def nn_architecture(image_size, parameter_dimension):
    #For distributed training with one machine and multiple gpus, collect cuda devices/gpus and then distribute model and
    #parameters to each of the gpus and train synchronously.
    strategy = tensorflow.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    #Distribute model to all gpus.
    with strategy.scope():

        #Build convolutional part of the nn. The convolutional part processes the spatial image and outputs a flatten vector.
        conv = models.Sequential()
        conv.add(layers.Conv2D(128, (3, 3), activation = 'relu', input_shape = (image_size, image_size, 2)))
        conv.add(layers.MaxPooling2D((2, 2), padding = "same"))
        conv.add(layers.Dropout(0.25))
        conv.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
        conv.add(layers.MaxPooling2D((2,2), padding = "same"))
        conv.add(layers.Dropout(0.25))
        conv.add(layers.Conv2D(16, (3, 3), activation = 'relu'))
        conv.add(layers.MaxPooling2D((2,2), padding = "same"))
        conv.add(layers.Dropout(0.5))
        conv.add(layers.Flatten())

        #Concatenate the parameters (2-d vector) to the flatten vector (64-d vector) produced from the convolutional part of the nn.
        #Concatenation results in a 66 dimension vector
        parameters_input = keras.layers.Input(shape = (parameter_dimension,))
        merged_output = layers.concatenate([conv.output, parameters_input])

        #This is the fully connected part of the nn which processes both the spatial image information and the parameter.
        model_combined = models.Sequential()
        model_combined.add(layers.Dense(64, input_shape = (66,)))
        model_combined.add(layers.Activation('relu'))
        model_combined.add(layers.Dense(16))
        model_combined.add(layers.Activation('relu'))
        model_combined.add(layers.Dense(8))
        model_combined.add(layers.Activation('relu'))
        model_combined.add(layers.Dense(2))
        model_combined.add(layers.Activation('softmax'))

        #This is the combined model (convolutional and full connected parts). The inputs are the spatial image (conv.input) and
        #the parameters (parameters_input) and the output is the output of model_combined.
        final_model = models.Model([conv.input, parameters_input], model_combined(merged_output))

        return final_model