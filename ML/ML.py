import tensorflow as tf
import tensorflow.keras as ks

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pydot
import graphviz

import os
import numpy as np
import matplotlib.pyplot as plt

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def plotImages(images_arr, label):
    fig, axes = plt.subplots(5, 5, figsize=(10,10))
    axes = axes.flatten()
    i = 0
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(class_names[label[i]])
        i = i+1
    plt.tight_layout()
    plt.savefig('MNIST.png')
    plt.clf()
    plt.close()
    return

def graphit(acc, loss, val_loss, model):
    domain = range(len(acc))

    plt.plot(domain, loss, 'bo', label= 'Training loss')
    plt.plot(domain, val_loss, 'rx', label= 'Test loss')
    plt.title('Training versus test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(str(model) + ".png")
    plt.clf()
    plt.close()
    return

def CONVNET(neurons):
    ConvNet = Sequential([
    Conv2D(64, (4,4), padding='same', activation='relu', input_shape=(28, 28,1)),
    MaxPooling2D((2,2)),
    Conv2D(32, (4,4), padding='same', activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(neurons, activation='relu'),
    Dense(10, activation='softmax')])

    return ConvNet

def NNTraining(model, x_train, y_train, xtest, ytest, num, output, batch, epoch, neurons = 0, newQuestion = False, loss1 = 'sparse_categorical_crossentropy'):

    model[0].compile(optimizer='adam',
          loss=loss1,
          metrics=['accuracy'])

    history = model[0].fit(
    x_train, 
    y_train, 
    validation_data=(xtest, ytest), 
    batch_size = batch, 
    epochs=epoch,  
    verbose=1)

    results = model[0].evaluate(xtest, ytest)

    graphit(history.history['accuracy'], history.history['loss'], history.history['val_loss'], model[1])

    ks.utils.plot_model(model[0],show_shapes=True, show_layer_names=True, to_file = model[1] + '_Network.png')

    if newQuestion:
        output.write("Start question " + str(num) + ":\n")
    output.write("Neurons in Dense Layer:" + str(neurons) + '\n')
    output.write("Model training Loss:\t" + str(history.history['loss'][-1]) + "\n")
    output.write("Model testing Loss:\t" + str(results[0]) + "\n")
    if newQuestion:
        if i==2:
            output.write('The training and testing loss curves have significant space between them indicating that this model is overfitting the data.\n')
        output.write("Model training accuracy:\t" + str(history.history['accuracy'][-1]) + "\n")
        output.write("Model testing  accuracy:\t" + str(results[1]) + "\n")
    if newQuestion == False:
        output.write("Model training accuracy:\t" +str(history.history['accuracy'][-1]) + "\n\n")
        output.write("Model testing  accuracy:\t" + str(results[1]) + "\n\n")
    if newQuestion:
        output.write("End question " + str(num) + ":\n\n")

    return (results[1],str(neurons))


### DEFINE THE PARAMETERS ###
(x_train, y_train), (xtest, ytest) = ks.datasets.fashion_mnist.load_data()
x_train, xtest = x_train/255.0, xtest/255.0
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
### DEFINE THE PARAMETERS ###

### Question 1 ###
plotImages(x_train[:25], y_train[:25])
### Question 1 ###

### DEFINE THE MODELS ###
#One layer of 500 neurons
Feed_Forward_Model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(500, activation='relu'),
    Dense(10, activation='softmax')])
#One layer of 500 neurons and L2 regularizations
Feed_Forward_Model_L2 = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(500, activation='relu', kernel_regularizer=ks.regularizers.l2(0.02)),
    Dense(10, activation='softmax')])
#5 Hidden Layers of 100 neurons
Feed_Forward_Model_DEEP = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(10, activation='softmax')])

#5 Hidden Layers of 100 neurons
Feed_Forward_Model_DEEP_L2 = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(500, activation='relu'),
    Dense(500, activation='relu'),
    Dense(500, activation='relu'),
    Dense(500, activation='relu'),
    Dense(500, activation='relu'),
    Dense(500, activation='relu'),
    Dense(500, activation='relu'),
    Dense(500, activation='relu'),
    Dense(500, activation='relu'),
    Dense(500, activation='relu'),
    Dense(500, activation='relu'),
    Dense(500, activation='relu'),
    Dense(500, activation='relu'),
    Dense(500, activation='relu'),
    Dense(500, activation='relu'),
    Dense(10, activation='softmax', kernel_regularizer=ks.regularizers.l2(0.02))])


output = open("MachineLearningHW6.txt", "w+")
output.write("Machine Learning HW 6.\n")
output.write("Travis Collier, Graduate Student.\n\n")
output.write("Problem 1: See attached image.\n\n")
ModelList = [(Feed_Forward_Model, 'FFN'), (Feed_Forward_Model_L2, 'FFN_L2'), (Feed_Forward_Model_DEEP, 'FFN_DEEP'), (Feed_Forward_Model_DEEP_L2, 'FFN_DEEP_L2')]
#### QUESTIONS 2:4 ###
i = 2
for model in ModelList:
    NNTraining(model, x_train, y_train, xtest, ytest, i ,output, 40, 10,newQuestion =True)
    i = i+1

output.close()

###
#HW7
###

output1 = open("MachineLearningHW7.txt", "w+")
output1.write("Machine Learning HW 7.\n")
output1.write("Travis Collier, Graduate Student.\n\n")

(x_train, y_train), (xtest, ytest) = ks.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(60000, 28, 28,1)
xtest = xtest.reshape(10000, 28, 28,1)
y_train = ks.utils.to_categorical(y_train, 10)
ytest = ks.utils.to_categorical(ytest, 10)
x_train, xtest = x_train / 255.0, xtest / 255.0
i=0
neurons = [20, 100, 300, 500, 800, 1000, 1500]
ModelList = [(CONVNET(neuron), "ConvNet" + str(neuron)) for neuron in neurons]
Best = (0.0, '10000')
for model in ModelList:
    curr = NNTraining(model, x_train, y_train, xtest, ytest, i+1 ,output1, 40, 3, neurons[i], False, 'categorical_crossentropy')
    if curr[0] > Best[0]:
        Best = curr
    i = i+1

output1.write("Question 3:\n")
output1.write("The best architecture had " + Best[1] + " neurons in the dense layer\n")
output1.write("The best architecture had accuracy " + str(Best[0]) + " \n")


### COMPILE THE MODELS ###