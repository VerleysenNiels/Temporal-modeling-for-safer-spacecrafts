import csv
import matplotlib.pyplot as plt
import numpy as np

"""Read predictions"""
def read(file):
    with open(file, "r") as infile:
        reader = csv.reader(infile)
        data = []
        for row in reader:
            data.append(row)
        return np.transpose(np.array(data))

"""Generates the plot of one training history file"""
def generate_plots(data, path):
    for i in range(0, int(len(data)/2)):
        real = data[i*2]
        predicted = data[(i*2)+1]

        name_real = real[0]
        name_predicted = predicted[0]
        signal_name = name_real.split(' ')[1]

        real_data = real[1:-1].astype('float64')
        predicted_data = predicted[1:-1].astype('float64')

        plt.plot(real_data)
        plt.plot(predicted_data)
        plt.ylabel("Signal value")
        plt.xlabel("Time (stepsize 0.01185 day)")
        plt.xlim(0, 100)
        plt.title('Real versus predicted ' + signal_name)
        plt.legend((name_real, name_predicted))
        plt.tight_layout()
        plt.savefig(path + signal_name, dpi=500)
        plt.clf()


if __name__ == '__main__':

    input_file = './Results/Predictions/6_LSTM_400_K_200_M_100.csv'
    output_folder = './Results/Predictions/Plots/6_LSTM_400_K_200_M_100/'

    data = read(input_file)
    generate_plots(data, output_folder)
