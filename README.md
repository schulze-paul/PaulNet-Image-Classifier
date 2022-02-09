# ML-image-classification

A machine learning with the CIFAR10 dataset.
The dataset consists of 60000 32x32 color images from 10 classes.

The classes are:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

I use a simple convolutional neural network with stochastic gradient descent for optimization. I run the algorithm on my GPU. The training is stopped early if network overfits on the training data set. When this occurs the validation loss starts to rise and training is stopped. 

---

## Network architecture and training

### Initial network architecture 

Two convolutional layers with a MaxPool layer in between and then three linear layers. I am using a learning rate scheduler with an initial learning rate of 0.05.

- image of layers

>validation loss: 1.41  
validation accuracy: 51.6%

This accuracy is already very high for such a simple network and shows that even simple solutions can be effective if the training is done correcty. The network clearly learned some of the features of the dataset. A classification by chance would be a 10% accuracy.

### Adding a second maxpool layer

Adding the maxpool layer after the second convolutional layer reduces the size of the first linear layer from 1400 to 350.

- image of layers


>validation loss: 1.39  
validation accuracy: 54.1%

The validation loss did not improve much, although accuracy did improve by 3% on the validation data set.

### Introducing ReLU activation functions

Adding a ReLu activation function sets all negative output values of the previous layer to zero, but avoids the vanishing gradient problem that occurs with convolutional layers and sigmoid and tanh activation functions.  

>validation loss: 1.31  
validation accuracy: 53.3%

The improvement in performance is offset somewhat by a longer training that resulted in a slight overfit.

### increasing the number of channels

Increasing the complexity of the model lead to a faster convergence and higher accuracy compared to the previous iteration of the model. 

>validation loss: 1.09  
validation accuracy: 62.0%

### Introducing BatchNorm

A BatchNorm layer as part of a composite convolutional layer normalizes the output batchwise. 

>validation loss: 0.996  
validation accuracy: 65.4%

### Adding a third composite layer

Further increasing the number of parameters and the complexity of the model, the third composite layer consists has the same composition as the other two composite layers. A convolutional layer, followed by a MaxPool, a ReLU activation function and a BatchNorm. Although the number of training steps needed for convergence has very consistent from the virst version to the last version, the computational compexity per training step has increased greatly because of the much higher number of trainable parameters.

>validation loss: 0.897  
>validation accuracy: 69.9%


### ReLU activation functions after the linear layers

Adding a ReLU activation function after the linear layers resulted in slight improvements in accuracy.

>validation loss: 0.880  
validation accuracy: 70.7%

### Adding a forth composite layer and increasing the amount of channels

---

- run on gpu
- using random crop transforms
- using random flip transforms
- normalizing the data

## initial network architecture

## training
- lr: 0.05
- using negative log likelyhood loss
- using stochastic gradient descent
- using learning rate scheduler

## improving the network architecture

1. using relu after convolutions
1. using batchnorm after convolution relus
1. using relus also after linear layers

## improving the training

- exploring the learing rate
- use adam instead of pure sgd
