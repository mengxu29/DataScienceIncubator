# The Problem
A description for the problem can be found here:
https://www.phmsociety.org/sites/phmsociety.org/files/PHM%20Data%20Challenge%202018%20vFinal%20v2_0.pdf

Enssentially, we are required to predict the remaining useful life (RUL).

The input is the time-series sensed data. 

The output is the time to next failure and the fault type.

The system is repariable. 
Also, the time when the system is shutdown for repair is recorded.

The difficulties may include:

1. imbalanced data. (The failure data is much less than the normal operating data).

2. feature extraction. Though it seems that the RNN do not need feature extraction, but it may be good to genearte some meaning features by hand.

3. what is the output? how to present the findings and outputs so that people will be impressed. (High/Low risk, RUL, IoT AWS deployment and visualization, ...)

![alt text](https://github.com/mengxu29/DataScienceIncubator/blob/master/Images/ds2.png)

# Very Useful Links
https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM

https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/cortana-analytics-playbook-predictive-maintenance

https://github.com/Azure/MachineLearningSamples-DeepLearningforPredictiveMaintenance

https://gallery.azure.ai/Notebook/Predictive-Maintenance-Modelling-Guide-R-Notebook-1#Problem-Description

https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#milling


# Data Cleaning and Preparation

## 1. Challenges
### 1.0 Nan values
Many nans exist in the data.

### 1.1 Missing timesteps
It is common to have missing observations from sequence data.

Data may be corrupt or unavailable, but it is also possible that your data has variable length sequences by definition. Those sequences with fewer timesteps may be considered to have missing values.

https://machinelearningmastery.com/handle-missing-timesteps-sequence-prediction-problems-python/

### 1.2 Very Long Sequences for LSTM
A reasonable limit of 250-500 time steps is often used in practice with large LSTM models. (Not sure. Maybe it's wrong.)

But, the data for the PHM could has a very long effect. The data is sampled every second. 200-500 time steps only cover a few minitues info, which is way not enough for inference. 

Special tricks should be took, e.g. https://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/

a) resample
b) average out the data
c) ...
### 1.3 Noise data (Sensor Noise), and outlier
Sensors may generate noise data, and even wired outliers. 

The outliers are hard to handle. It may be important indicators for failure, or it just the outliered noise.

People arealdy notice the noise will affect the performance. (One paper metioned, not confirmed)

### 1.4 Few machines (The model learned from data of one machine is good for another machine?)
The data is collected only from 22 machines. Although each machine generates a lot of data, the data considering machine difference is too less, and it is still challengable to generalize the model learned from these few machines to a totally new machine. 

### 1.5 Data transformatoin 
Data Preparation
Before we can fit a model to the dataset, we must transform the data.

The following three data transforms are performed on the dataset prior to fitting a model and making a forecast.

Transform the time series data so that it is stationary. Specifically, a lag=1 differencing to remove the increasing trend in the data.

Transform the time series into a supervised learning problem. Specifically, the organization of data into input and output patterns where the observation at the previous time step is used as an input to forecast the observation at the current time step

Transform the observations to have a specific scale. Specifically, to rescale the data to values between -1 and 1.
These transforms are inverted on forecasts to return them into their original scale before calculating and error score.

https://machinelearningmastery.com/use-dropout-lstm-networks-time-series-forecasting/

### 1.6 New Data and Retrain the model
Every time when a new machine is used, the data may have a very different distribution. 
Then it maybe need a retraining for the mdoel

https://medium.com/ibm-watson-data-lab/keeping-your-machine-learning-models-up-to-date-f1ead546591b

## 2. Data Preparation: Series to Supervised
Shift dataset

https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/

## 3. Keras LSTM
Some tips to help you when preparing your input data for LSTMs.

The LSTM input layer must be 3D.

The meaning of the 3 input dimensions are: samples, time steps, and features.

The LSTM input layer is defined by the input_shape argument on the first hidden layer.

The input_shape argument takes a tuple of two values that define the number of time steps and features.

The number of samples is assumed to be 1 or more.

The reshape() function on NumPy arrays can be used to reshape your 1D or 2D data to be 3D.

The reshape() function takes a tuple as an argument that defines the new shape.

https://machinelearningmastery.com/prepare-univariate-time-series-data-long-short-term-memory-networks/

and

https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/

# 1. Models
The review paper gives a good summary of the models for PHM

https://github.com/ClockworkBunny/MHMS_DEEPLEARNING

Plan to implement two models. 
## Recurrent Neural Network (RNN)
RNN can handle sequential data very well, (e.g. https://towardsdatascience.com/recurrent-neural-networks-and-lstm-4b601dd822a5, and https://en.wikipedia.org/wiki/Recurrent_neural_network)

RNN is very powerful for many tasks: speech recognition, translation, text processing, natural language processing (NLP). The big company actually use it for many products, like apple siri, google voice, ...

Mainly there are two popular RNN model: GRU and LSTM

## CNN
... do it later

# 2. Codes
Keras is a very popular library for deep learning. https://keras.io/why-use-keras/

The code is based on Keras.


## Install required packages
In anocoda,
Install Keras and Tensorflow

## Use Keras RNN to train
https://keras.io/layers/recurrent/


# 3. Presentation
It's important to present the output well so that people can be impressed.
## Model explanation 

## Data visualization
https://github.com/Azure/MachineLearningSamples-DeepLearningforPredictiveMaintenance/blob/master/Code/1_data_ingestion_and_preparation.ipynb

## IoT AWS demo
