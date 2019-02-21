# The Problem
A description for the problem can be found here:
https://www.phmsociety.org/sites/phmsociety.org/files/PHM%20Data%20Challenge%202018%20vFinal%20v2_0.pdf

Predictive maintenance is a popular application of predictive analytics that can help business in several industries achieve high asset
utilization and savings in operational costs.

The goal of predictive maintenance is to predict when an asset may fail in the near future, and to estimate the remaining life (or time-to-failure) of an asset based on sensor data collected overtime to monitor the state of an equipment. 
Preventive maintenance can extend component lifespans and reduce unscheduled maintenance and labor costs, businesses can gain cost savings and competitive advantages.

This proposed project will predict the fault behavior of an ion mill etch tool used in a wafer manufacturing process.

The model uses sensor data to predict when the machine will fail in the future so that maintenance can be planned in advance. The question to ask is “given these machine operation and failure events history, can we predict when the machine will fail?” we re-formulate this question into two closely relevant questions and answer them using two different types of machine learning models:
1. Supervised regression model: how long a machine will last before it fails?
2. Binary classification model: is this machine going to fail within one week? (failing: high risk; not failing: low risk)

# Datasets

In the dataset directory there are training, validation and testing datasets. The data consists of multiple multivariate time series with “seconds” as the time unit, together with 22 sensor readings for each time. The following pictures show a sample of the data:

![alt text](https://github.com/mengxu29/DataScienceIncubator/blob/master/pic/sample1.jpg)

![alt text](https://github.com/mengxu29/DataScienceIncubator/blob/master/pic/sample2.png)

## Data visualization
1000 time series data with 22 sensor readings
<p align="center"> 
<img src="https://github.com/mengxu29/DataScienceIncubator/blob/master/pic/visualization.jpg">
</p>

## Data cleaning and preprocessing

### Remove null data

### Resample of long time series data
The data is sampled every second. Resample time series data from one second to ten seconds.

### Data transformatoin 

Transform the time series into a supervised learning problem, where the observation at the previous time step is used as an input to forecast the observation at the current time step. Transform the observations to have a specific scale. Specifically, to rescale the data to values between -1 and 1. These transforms are inverted on forecasts to return them into their original scale before calculating and error score.

# Models
## Gated Recurrent Unit (GRU)

Recurrent Neural Network (RNN) can handle sequential data very well, RNN is very powerful for many tasks: speech recognition, translation, text processing, natural language processing (NLP). The big company actually use it for many products, like apple siri, google voice, etc. Mainly there are two popular RNN model: Gated Recurrent Unit (GRU) and Long Short Term Memory (LSTM)

I build a Gated Recurrent Unit (GRU) network to predict remaining useful life (or time-to-failure) of machine and its risk status.
GRU network is especially appealing to the predictive maintenance due to its ability at learning time series data.

<p align="center"> 
<img src="https://github.com/mengxu29/DataScienceIncubator/blob/master/pic/diagram.png">
</p>

# Codes
Keras is a very popular library for deep learning. The code is based on Keras with Tensorflow as the back end.

## Results of training
Based on training datasets, the following pictures show the trend of loss function and accuracy.

<p align="center"> 
<img src="https://github.com/mengxu29/DataScienceIncubator/blob/master/pic/loss.jpg">
</p>

<p align="center"> 
<img src="https://github.com/mengxu29/DataScienceIncubator/blob/master/pic/accuracy.png">
</p>

## Results of prediction

<p align="center"> 
<img src="https://github.com/mengxu29/DataScienceIncubator/blob/master/pic/prediction.png">
</p>

<p align="center"> 
<img src="https://github.com/mengxu29/DataScienceIncubator/blob/master/pic/prediction%20risk.jpg">
</p>

## Learned features

<p align="center"> 
<img src="https://github.com/mengxu29/DataScienceIncubator/blob/master/pic/feature.jpg">
</p>
