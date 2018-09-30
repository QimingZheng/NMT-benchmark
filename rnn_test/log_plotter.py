import matplotlib.pyplot as plt
import numpy as np

def read_log(filename):
    time_stamp = []
    cross_entro = []
    with open(filename) as file:
        text = file.readlines()
        for i in range(len(text)):
            text[i] = text[i].split()
            time_stamp.append(int(text[i][0]))
            cross_entro.append(float(text[i][1]))
    return time_stamp, cross_entro

def plot(time_stamp, sequence):
#    time_stamp = np.array(time_stamp)
#    sequence = np.array(sequence)
    plt.plot(time_stamp, sequence, 'ro')
    plt.grid()
    plt.xlabel("Batches")
    plt.ylabel("cross entropy loss")
    plt.savefig("batch-loss.png")
    return

time_stamp, sequence = read_log("train.log")
plot(time_stamp, sequence)
