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

![alt text](https://github.com/mengxu29/DataScienceIncubator/blob/master/ds2.png)

# Very Useful Links
https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM

https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/cortana-analytics-playbook-predictive-maintenance

https://github.com/Azure/MachineLearningSamples-DeepLearningforPredictiveMaintenance

https://gallery.azure.ai/Notebook/Predictive-Maintenance-Modelling-Guide-R-Notebook-1#Problem-Description

https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#milling
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
