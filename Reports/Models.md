# 1. LSTM
understand LSTM: https://colah.github.io/posts/2015-08-Understanding-LSTMs/


## several 1-dimensional convolutions before LSTM
https://www.tensorflow.org/tutorials/sequences/recurrent_quickdraw

# 2. Model Challenge
## Avoid overfitting
The model can be easily overfitted. It's especially a difficult probelm here. 

To avoid overfitting, there are some tips:

### 2.1 Reduce the network’s capacity
by removing layers or reducing the number of elements in the hidden layers

### 2.2 Apply regularization, 
which comes down to adding a cost to the loss function for large weights

### 2.3 Use Dropout layers, 
which will randomly remove certain features by setting them to zero.

Add Dropout between LSTM layers to reduce the overfitting. What is the function of dropout between LSTM layers?
https://www.tensorflow.org/tutorials/sequences/recurrent
### 2.4 other methods
Add more data

Use data augmentation

Use architectures that generalize well

https://towardsdatascience.com/handling-overfitting-in-deep-learning-models-c760ee047c6e
