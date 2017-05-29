import AEDAT_Handler as f
import neuralNet as n
import pyNN.spiNNaker as p
import random as r

basepath = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/TrainingSamples"
traininDirs = [basepath]

timeBetweenSamples = 100

Network = n.NeuralNet()


