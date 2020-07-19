# Temporal-modeling-for-safer-spacecrafts
Main experiment of my thesis: Temporal modeling for safer spacecrafts. In this repository I use an LSTM network for forecasting of multiple time series coming from spacecraft sensors. I am using a subset of sensors from the Mars Express orbiter. By thresholding the loss of this predictor we can determine if the measurements from the spacecraft are anomalous or not. The system can also provide a ranking of the sensors based on their loss, this gives an indication of which sensors are probably anomalous.

In the results folder the predictions of the sensors are shown under Predictions and some plots showing the anomaly detection at work are shown under Anomaly.
For more information you can always ask me questions or you can read the thesis text by requesting it at KU Leuven (or ask me), alternatively you can take a look at [the little brother](https://github.com/VerleysenNiels/LSTM_Periodic_Function) of this project.
