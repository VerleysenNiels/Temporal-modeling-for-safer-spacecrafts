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
import matplotlib.pyplot as plt

"""
Define global variables
"""
EPOCHS = 500
BATCH_SIZE = 200
TRAINING_SIZE = 10000
NAME = "6_LSTM_300_K_200_M_100"    # Output files will be saved with this name in their respective folders
K = 200                              # Number of previous timesteps as input
M = 100                              # Number of timesteps predicted
architecture_LSTM = [300, 300, 300, 300, 300, 300]  # Define LSTM layers of the network
architecture_FC = []                 # Define Fully Connected layers of the network
architecture_CNN = []       #[[2800, 4], [1400, 3]]                # Define CNN layers of the network

END = 1000      # 10 days
anomaly_name = "Anomaly_NDWDBT0K" # NACW0S00, NACW0S01, NDMA5790, NDWDBT0K"
anomalous = [12]# 0, 2, 3, 12]
start_times = [200]

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
MAIN for training and testing a neural network on the dataset
"""
def train_and_test():
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


"""
MAIN for testing anomaly detection
"""
def test_anomaly():
    # SETUP
    dataset_expt = Dataset_Exporter()
    dataset_expt.load("./Data/Dataset.pickle")
    dataset_expt.normalize(TRAINING_SIZE)

    # Add anomalies to the data
    dataset_expt.add_anomaly(TRAINING_SIZE, TRAINING_SIZE+1001, -0.01, anomalous, start_times)

    S = len(dataset_expt.dataset.sensors)

    # Build model
    model = LSTM_network(S, K, M, architecture_LSTM, architecture_CNN=architecture_CNN, architecture_FC=architecture_FC)

    # Load model
    model.load("./Results/Trained/6_LSTM_300_K_200_M_100.hdf5")

    # Setup sensor ranking and predictions
    mae = []
    predictions = [] # Structured as [time1, time2, time3, ...] with timex = [sensor1, sensor2, ...] where each sensor contains the predictions for that sensor at that timestep
    for sensor in dataset_expt.dataset.sensors:
        mae.append([])

    # Loop
    for time in range(0, END):
        # Predict next day
        x = np.transpose(np.array(dataset_expt.dataset.normalized[time+TRAINING_SIZE-K: time+TRAINING_SIZE]))
        x = np.expand_dims(x, axis=0)
        yhat = model.predict(x)

        # Add predictions
        # Loop over the sensors
        for i in range(0, len(dataset_expt.dataset.sensors)):
            # Loop over predicted timesteps
            for t in range(0, M):
                # Add new list for predictions if this timestep does not have one
                if len(predictions) < time+t+1:
                    new_list = []
                    for s in dataset_expt.dataset.sensors:
                        new_list.append([])
                    predictions.append(new_list)
                # Add prediction to the right list
                predictions[time+t][i].append(yhat[0][i*M + t])

        # Determine absolute error of each prediction for this timestep
        y = np.transpose(np.array(dataset_expt.dataset.normalized[time + TRAINING_SIZE]))
        for s in range(0, len(dataset_expt.dataset.sensors)):
            nr = 0
            error = 0
            for pred in predictions[time][s]:
                difference = abs(y[s] - pred)
                error += difference
                nr += 1
            e = error / nr
            mae[s].append(e)

    # Plot the mean absolute error
    errors = []
    for t in range(0, len(mae[0])):
        error = 0
        nr = 0
        for s in range(0, len(mae)):
            if s not in [8, 10, 11]:        # Remove NACAH signals
                error += mae[s][t]
                nr += 1
        e = error / nr
        errors.append(e)

    plt.plot(errors)
    plt.ylabel("MAE")
    plt.xlabel("Time (stepsize 0.01185 day)")
    plt.xlim(0, END)
    plt.ylim(-0.2, 1)
    plt.title('Loss over time')
    plt.tight_layout()
    plt.savefig("./Results/Anomaly/" + anomaly_name, dpi=500)
    plt.clf()

    names = dataset_expt.dataset.sensors
    index = list(range(0, len(mae)))
    index.pop(11)
    index.pop(10)
    index.pop(8)

    # Print ordering of sensors at different timesteps
    print("Top 5 sensors according to loss:")
    for t in [50, 200, 210, 220, 230, 240, 250, 300, 500, 510, 520, 530, 540, 550, 600]:
        ordering = sorted(index, reverse=True, key=lambda s: mae[s][t])
        print("At timestep " + str(t) + ":")
        for i in range(0, 5):
            print("\t" + str(i+1) + ". " + names[ordering[i]] + "\t" + str(mae[ordering[i]][t]))

    false_positives = []
    delays = []
    start = start_times[0]
    thresholds = np.arange(0, 2, 0.05)

    # For each threshold -> get detected anomalies
    for threshold in thresholds:
        fp = 0
        delay = 0
        first = True
        for i in range(0, len(errors)):
            # Anomaly detected?
            if errors[i] > threshold:
                # False detection?
                if i + 1 < start:
                    fp += 1
                elif first:
                    delay = i - start + 1
                    first = False
        if first:
            delays.append(None)
        else:
            delays.append(delay)
        false_positives.append(fp / start)

    # Plot detection delay and false positive rate for different thresholds
    plt.plot(thresholds, false_positives)
    plt.xlabel("Threshold")
    plt.ylabel("False positive rate")
    plt.savefig("./Results/Anomaly/Plot_thresh-fpr_" + anomaly_name, dpi=500)
    plt.clf()

    plt.plot(false_positives, delays)
    plt.xlabel("False positive rate")
    plt.ylabel("Delay")
    plt.savefig("./Results/Anomaly/Plot_fpr-delay_" + anomaly_name, dpi=500)
    plt.clf()

"""
MAIN
"""
if __name__ == '__main__':

    # train_and_test()

    test_anomaly()
