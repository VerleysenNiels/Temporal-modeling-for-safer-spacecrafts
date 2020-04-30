#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:08:53 2020

@author: Niels Verleysen

Class that manages the LSTM network for predicting the satellite data.
You first have to build the model by specifying the model architecture.
Then you can train the model on given data and use this model to predict M steps into the future.

S = number of sensors
K = number of previous points visible
M = amount of timesteps predicted into the future
"""

from keras.models import Model
from keras.layers import LSTM, Input, Dense, Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import losses
import numpy as np
import csv

class LSTM_network:
    
    def __init__(self, s, k, m, architecture_LSTM, architecture_CNN = [], architecture_FC = []):
        inputs = Input(shape=(s, k))
        l = inputs

        if len(architecture_CNN) > 0:
            for layer in architecture_CNN:
                l = Conv1D(layer[0], layer[1])(l)
                l = MaxPooling1D()(l)

        for i in range(0, len(architecture_LSTM)-1):
            l = LSTM(int(architecture_LSTM[i]), return_sequences=True)(l)

        l = LSTM(int(architecture_LSTM[-1]), return_sequences=False)(l)

        if len(architecture_FC) > 0:
            for layer in architecture_FC:
                l = Dense(layer)(l)

        outputs = Dense(s*m)(l) # M values for each of S sensors
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss=losses.mean_absolute_error, optimizer='Adam', metrics=['accuracy', 'mae'])
        
        self.model.summary()
        
        filepath="./Results/Weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        self.callbacks_list = [checkpoint]
        
    def train(self, X, Y, epochs, batch_size):
        history = self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, callbacks=self.callbacks_list)
        return history
    
    def load(self, file):
        self.model.load_weights(file)
    
    def predict(self, previous_vals):
        prediction = self.model.predict(previous_vals, verbose=0)
        return prediction

    """ 
        Evaluation of the model on given sensor data.
        Expects a set of input-output pairs and returns the average loss and the standard deviation of the loss
        It also outputs a csv file with all predictions, allowing us to make a plot afterwards
        
        x is a 2D-array with k values for every sensor (used as input)
        y is a 2D-array with m values for every sensor (model will try to predict these)
        sensors is a list of sensor names
    """
    def evaluate_and_write_m(self, x, y, file, s, m, sensors):
        losses = []
        stdvs = []
        with open(file, 'w', newline='') as outfile:
            w = csv.writer(outfile)
            title = []
            for sensor in sensors:
                title.append('Real ' + str(sensor))
                title.append('Predicted ' + str(sensor))
            w.writerow(title)

            # Make single prediction
            yhat = self.model.predict(np.expand_dims(x, axis=0))

            predictions = []

            # Process prediction
            for sensor in range(0, s):
                sensor_losses = []
                sensor_predictions = [[],[]] #[0] is filled with real data, [1] is filled with the predictions
                for value in range(0,m):
                    sensor_losses.append(abs(yhat[0][sensor*m + value] - y[sensor][value]))
                    sensor_predictions[0].append(y[sensor][value])
                    sensor_predictions[1].append(yhat[0][sensor*m + value])
                losses.append(np.mean(np.array(sensor_losses)))
                stdvs.append(np.std(np.array(sensor_losses)))
                predictions.append(sensor_predictions[0])
                predictions.append(sensor_predictions[1])

            predictions = np.transpose(np.array(predictions))
            for row in predictions:
                w.writerow(row)

        return losses, stdvs
