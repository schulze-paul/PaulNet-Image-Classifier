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


> validation loss: 1.39  
> validation accuracy: 54.1%

The validation loss did not improve much, although accuracy did improve by 3% on the validation data set.

### Introducing ReLU activation functions

Adding a ReLu activation function sets all negative output values of the previous layer to zero, but avoids the vanishing gradient problem that occurs with convolutional layers and sigmoid and tanh activation functions.  

> validation accuracy: 53.3%
> validation loss: 1.31  

The improvement in performance is offset somewhat by a longer training that resulted in a slight overfit.

### increasing the number of channels

Increasing the complexity of the model lead to a faster convergence and higher accuracy compared to the previous iteration of the model. 

> validation accuracy: 62.0%  
> validation loss: 1.09  

### Introducing BatchNorm

A BatchNorm layer as part of a composite convolutional layer normalizes the output batchwise. 

> validation accuracy: 65.4%  
> validation loss: 0.996  

### Introducing a third composite layer

Further increasing the number of parameters and the complexity of the model, the third composite layer consists has the same composition as the other two composite layers. A convolutional layer, followed by a MaxPool, a ReLU activation function and a BatchNorm. Although the number of training steps needed for convergence has very consistent from the virst version to the last version, the computational compexity per training step has increased greatly because of the much higher number of trainable parameters.

> validation accuracy: 69.9%  
> validation loss: 0.897  


### ReLU activation functions after the linear layers

Adding a ReLU activation function after the linear layers resulted in slight improvements in accuracy.

> validation accuracy: 70.7%  
> validation loss: 0.880  

### Increasing the amount of channels further 

This resulted in even higher accuracy. The risk with adding complexity to the model is that it can start overfitting on the train/val dataset, which can lower performance on the test set.

> validation accuracy: 72.4%  
> validation loss: 0.868  

### Introducing crop and flip transformations

I introduced random crop and horizontal flip transformations on the train/val dataset in order to increase the sample size. This lead to a higher accuracy on the validation dataset and a dramatically lower validation loss. It also helped reduce overfitting which meant more training steps were possible before the validation loss went up again.

> validation accuracy: 76.6%  
> validation loss: 0.672  

### Introducing a forth composite layer and increasing the number of channels

I also had to lower the kernel size for the convolutional layers to 2x2.
This increase in complexity came with another slight boost in performance.

> validation accuracy: 77.6%  
> validation loss: 0.645

### Introducing patience to the early stopping mechanism

I had a suspicion that the training was stopping _too_ early, so I added a patience term of 3. This brought the accuracy over the 80% mark.

> validation accuracy: 80.6%  
> validation loss: 0.483

---

## Performance on test set

In the end I tested the performance of the neural network on the test dataset. This was the first and only time I used the test set for this model. It represents the performance of the model on data that it did not come in contact with and also shows that the models ability to generalize is excellent.

> test accuracy: 80.6%  
> test loss: 0.574

The model generalizes well and did not overfit on the train/val dataset. Performance as measured by the accuracy is almost identical to the performance on the validation set.  
The model has 215,710 parameters.

---

## Comparison with Resnet18 

Resnet is a convolutional neural network with 18 layers. It has about 11 Million parameters, so about 50x as many as I have trained. I modified it slightly to make it work with the 32x32 image dimensions of CIFAR10.

> validation accuracy: 83.6%  
> validation loss: 0.377

> test accuracy: 81%  
> test loss: 0.627

As expected, the accuracy of Resnet is higher. However I was able to reach to a comparable accuracy with a much simpler architecture. 
The validation loss is much lower than what I archieved in my model though.


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
