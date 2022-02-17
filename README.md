# PaulNet-Image-Classifier

Author: [Paul Schulze](https://schulze-paul.github.io)  
Task: 🖼️ Image classification    
Dataset: 💾 [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)  
Test accuracy: 🎯 [80.6%](#-performance-on-the-test-set)

---


In this project, I use my experiences from [Prof. Niessner](https://www.niessnerlab.org/members/matthias_niessner/profile.html)s course [I2DL](https://niessner.github.io/I2DL/) at [TUM](https://www.tum.de/) to improve a very simple convolutional neutral network (CNN) and bring the validation accuracy from ~50% to ~80%. The CNN is trained on an image classification task on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and reaches a test accuracy of 80.6%.

- [Dataset](#-dataset-description) 
- [Performance](#-performance-on-the-test-set) 
- [Architecture](#%EF%B8%8F-network-architecture) 
- [Improvements](#-network-improvements) 
- [Comparison with Resnet18](#-comparison-with-resnet18)

## 📰 Dataset Description

Bird             | Car             | Cat             | Deer             | Dog             | Frog             | Horse             | Plane             | Ship             | Truck
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/schulze-paul/PaulNet-Image-Classifier/blob/main/images/Examples/bird.png?raw=true" width=100>  |  <img src="https://github.com/schulze-paul/PaulNet-Image-Classifier/blob/main/images/Examples/car.png?raw=true" width=100>|  <img src="https://github.com/schulze-paul/PaulNet-Image-Classifier/blob/main/images/Examples/cat.png?raw=true" width=100>|  <img src="https://github.com/schulze-paul/PaulNet-Image-Classifier/blob/main/images/Examples/deer.png?raw=true" width=100>|  <img src="https://github.com/schulze-paul/PaulNet-Image-Classifier/blob/main/images/Examples/dog.png?raw=true" width=100>|  <img src="https://github.com/schulze-paul/PaulNet-Image-Classifier/blob/main/images/Examples/frog.png?raw=true" width=100>|  <img src="https://github.com/schulze-paul/PaulNet-Image-Classifier/blob/main/images/Examples/horse.png?raw=true" width=100>|  <img src="https://github.com/schulze-paul/PaulNet-Image-Classifier/blob/main/images/Examples/plane.png?raw=true" width=100>|  <img src="https://github.com/schulze-paul/PaulNet-Image-Classifier/blob/main/images/Examples/ship.png?raw=true" width=100>|  <img src="https://github.com/schulze-paul/PaulNet-Image-Classifier/blob/main/images/Examples/truck.png?raw=true" width=100>
<p align='center'>Table 1: Example images for each class.</p>

CIFAR-10 Consists of 60 K annotated images. Each class has 5 K images in the training / validation split. 

<table align='center'> 
  <tr>
    <td align='center'><b>Split</b></td>
    <td align='center'><b># Images</b></td>
    <td align='center'><b># Images per class</b></td>
  </tr>
  <tr>
    <td align='center'>Train / Val</td>
    <td align='center'>50,000</td>
    <td align='center'>5,000</td>
  </tr>
  <tr>
    <td align='center'>Test</td>
    <td align='center'>10,000</td>
    <td align='center'>1,000</td>
  </tr>
</table>
<p align='center'>Table 2: Dataset stats.</p>




## 🔥 Performance on the Test Set

- test accuracy: 80.6%  
- test loss: 0.574

The performance on the test dataset shows that the model is able to generalize from the training and validation dataset to unseen data. The accuracy is close to the performance of the Resnet18 network, which has much higher complexity.
The model generalizes well and did not overfit on the train/val dataset. Performance as measured by the accuracy is almost identical to the performance on the validation set.  


## ⚙️ Network Architecture

![nn architecture](https://github.com/schulze-paul/PaulNet-Image-Classifier/blob/main/images/nn_architecture.png?raw=true)
**Figure 1: Neural Network Architecture.** The network has four composite layers and three linear layers. 


The network has four composite convolutional layers. Each of these layers is a combination of a convolutional layer with a max pool layer, a ReLU activation function, and a batch norm layer. It has a total of 215,710 parameters.


<table align='center'> 
  <tr>
    <td><b>Layer</b></td>
    <td><b>Size</b></td>
    <td><b>Channels</b></td>
    <td><b>Kernel</b></td>
    <td><b>Stride</b></td>
  </tr>
  <tr>
    <td align=>Input</td>
    <td align='right'>32x32</td>
    <td align='right'>3</td>
    <td align='right'> </td>
    <td align='right'> </td>
  </tr>
  <tr>
    <td align=>Convolutional</td>
    <td align='right'>31x31</td>
    <td align='right'>32</td>
    <td align='right'>2x2</td>
    <td align='right'>1x1</td>
  </tr>
 
  <tr>
    <td align=>MaxPool</td>
    <td align='right'> </td>
    <td align='right'> </td>
    <td align='right'>2x2</td>
    <td align='right'>1x1</td>
  </tr>
 
  <tr>
    <td align=>ReLU</td>
    <td align='right'> </td>
    <td align='right'> </td>
    <td align='right'> </td>
    <td align='right'> </td>
  </tr>
 
  <tr>
    <td align=>BatchNorm</td>
    <td align='right'> </td>
    <td align='right'>64</td>
    <td align='right'> </td>
    <td align='right'> </td>
  </tr>
 
  <tr>
    <td align=>Output</td>
    <td align='right'>15x15</td>
    <td align='right'>32</td>
    <td align='right'> </td>
    <td align='right'> </td>
  </tr>
 
</table>
<p align='center'>Table 3: First Composite Layer.</p>

## 📈 Network Improvements

I started with a very simple network and optimization algorithm and gradually increased the complexity of both, which resulted in gradual increases in performance.

![validation accuracy](https://github.com/schulze-paul/ML-image-classification/blob/main/images/val_acc_grey.png?raw=true)
**Figure 2: Validation Accuracy.** As I improved the network architecture and the training algorithm, the network reached higher and higher accuracy. For comparison the performance of my modified version of  Resnet18 is also shown. Data imported from tensorboard.


![validation loss](https://github.com/schulze-paul/ML-image-classification/blob/main/images/val_loss_grey.png?raw=true)
**Figure 3: Validation Loss.** As I improved the network architecture and the training algorithm, The validation loss decreased further and further. For comparison the performance of my modified version of Resnet18 is also shown. Data imported from tensorboard.

#### Initial network architecture 
> validation accuracy: 51.6%  
> validation loss: 1.41  

Two convolutional layers with a MaxPool layer in between and then three linear layers. I am using a learning rate scheduler with an initial learning rate of 0.05. 
This accuracy is already very high for such a simple network and shows that even simple solutions can be effective if the training is done correcty. The network clearly learned some of the features of the dataset. A classification by chance would be a 10% accuracy.

#### Adding a second maxpool layer
> validation accuracy: 54.1%  
> validation loss: 1.39  

Adding the maxpool layer after the second convolutional layer reduces the size of the first linear layer from 1400 to 350.
The validation loss did not improve much, although accuracy did improve by 3% on the validation data set.

#### Introducing ReLU activation functions
> validation accuracy: 53.3%
> validation loss: 1.31 

Adding a ReLu activation function sets all negative output values of the previous layer to zero, but avoids the vanishing gradient problem that occurs with convolutional layers and sigmoid and tanh activation functions.  
The improvement in performance is offset somewhat by a longer training that resulted in a slight overfit.

#### increasing the number of channels
> validation accuracy: 62.0%  
> validation loss: 1.09 

Increasing the complexity of the model lead to a faster convergence and higher accuracy compared to the previous iteration of the model. 

#### Introducing BatchNorm
> validation accuracy: 65.4%  
> validation loss: 0.996  

A BatchNorm layer as part of a composite convolutional layer normalizes the output batchwise. 

#### Introducing a third composite layer
> validation accuracy: 69.9%  
> validation loss: 0.897  

Further increasing the number of parameters and the complexity of the model, the third composite layer consists has the same composition as the other two composite layers. A convolutional layer, followed by a MaxPool, a ReLU activation function and a BatchNorm. Although the number of training steps needed for convergence has very consistent from the virst version to the last version, the computational compexity per training step has increased greatly because of the much higher number of trainable parameters.

#### ReLU activation functions after the linear layers
> validation accuracy: 70.7%  
> validation loss: 0.880  

Adding a ReLU activation function after the linear layers resulted in slight improvements in accuracy.

#### Increasing the amount of channels further 
> validation accuracy: 72.4%  
> validation loss: 0.868  

This resulted in even higher accuracy. The risk with adding complexity to the model is that it can start overfitting on the train/val dataset, which can lower performance on the test set.

#### Introducing crop and flip transformations
> validation accuracy: 76.6%  
> validation loss: 0.672  

I introduced random crop and horizontal flip transformations on the train/val dataset in order to increase the variance in the train/validation set. This lead to a higher accuracy on the validation dataset and a dramatically lower validation loss. It also helped reduce overfitting which meant more training steps were possible before the validation loss went up again.

#### Introducing a forth composite layer and increasing the number of channels
> validation accuracy: 77.6%  
> validation loss: 0.645

I also had to lower the kernel size for the convolutional layers to 2x2.
This increase in complexity came with another slight boost in performance.

#### Introducing patience to the early stopping mechanism
> validation accuracy: 80.6%  
> validation loss: 0.483

I had a suspicion that the training was stopping _too_ early, so I added a patience term of 3. This brought the accuracy over the 80% mark.

## 🔬 Comparison with Resnet18 
> validation accuracy: 83.6%  
> validation loss: 0.377

> test accuracy: 81%  
> test loss: 0.627

Resnet is a convolutional neural network with 18 layers. It has about 11 Million parameters, so about 50x as many as I have trained. I modified it slightly to make it work with the 32x32 image dimensions of CIFAR10. The training and optimization algorithm is the same that I used for my network. I am sure it is possible to get even higher performance from this network with a better optimization algorithm.
As expected, the accuracy of Resnet is higher. However I was able to reach to a comparable accuracy with a much simpler architecture. 
The validation loss is much lower than what I archieved in my model though.
