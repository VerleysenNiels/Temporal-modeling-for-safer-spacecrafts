#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:15:11 2020

@author: Niels Verleysen

Main script for the experiment.

This script makes use of the Dataset Exporter script from the repository with the same name and requires a dataset file created by that repository.

1) Load and normalize dataset
2) Build specified model
3) Train model on the dataset
4) Predict M steps into the future, write these predictions to a csv file and write the loss + standard deviation of the loss for each sensor to another csv file
"""

import pickle
from LSTM_network import LSTM_network
from Dataset_Exporter import Dataset_Exporter, Dataset
import numpy as np
import csv

"""
Define global variables
"""
EPOCHS = 500
BATCH_SIZE = 200
TRAINING_SIZE = 10000
NAME = "7_LSTM_200_K_200_M_100"    # Output files will be saved with this name in their respective folders
K = 200                              # Number of previous timesteps as input
M = 100                              # Number of timesteps predicted
architecture_LSTM = [200, 200, 200, 200, 200, 200, 200]  # Define LSTM layers of the network
architecture_FC = []                 # Define Fully Connected layers of the network
architecture_CNN = []                # Define CNN layers of the network

"""
Build training set 
"""
def build_dataset(k, m, dataset_expt):

    X = []
    Y = []

    for step in range(0, TRAINING_SIZE+1):
        X.append(np.transpose(np.array(dataset_expt.dataset.normalized[step:step+k])))
        Y.append((np.transpose(np.array(dataset_expt.dataset.normalized[step+k:step+k+m]))).flatten())
    
    return np.array(X), np.array(Y)

"""
MAIN
"""
if __name__ == '__main__':

    dataset_expt = Dataset_Exporter()
    dataset_expt.load("./Data/Dataset.pickle")
    dataset_expt.normalize(TRAINING_SIZE)

    S = len(dataset_expt.dataset.sensors)

    # Build training set
    X, Y = build_dataset(K, M, dataset_expt)

    # Build model
    model = LSTM_network(S, K, M, architecture_LSTM, architecture_CNN=architecture_CNN, architecture_FC=architecture_FC)

    # Train model
    history = model.train(X, Y, EPOCHS, BATCH_SIZE)

    # Save this progress
    model.model.save("./Results/Trained/" + str(NAME) + ".hdf5")
    with open("./Results/Training_history/" + str(NAME) + ".pickle", 'wb') as outfile:
        pickle.dump(history, outfile)

    # Test the model
    X = np.transpose(np.array(dataset_expt.dataset.normalized[TRAINING_SIZE+1 : TRAINING_SIZE + 1 + K]))
    Y = np.transpose(np.array(dataset_expt.dataset.normalized[TRAINING_SIZE + 2 +K : TRAINING_SIZE + 2 + K + M]))
    losses, stdvs = model.evaluate_and_write_m(X, Y, "./Results/Predictions/" + str(NAME) + ".csv", S, M, dataset_expt.dataset.sensors)

    # Write losses to a csv file
    with open("./Results/Losses/" + str(NAME) + ".csv", 'w', newline='') as outfile:
        w = csv.writer(outfile)
        w.writerow(["Sensor", "Mean Absolute Error", "Standard Deviation of the loss"])
        for i in range(0, len(losses)):
            w.writerow([dataset_expt.dataset.sensors[i], losses[i], stdvs[i]])
