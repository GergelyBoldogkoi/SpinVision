import AEDAT_Handler as f
import neuralNet as n
import pyNN.spiNNaker as p
import random as r

basepath = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/TrainingSamples"
traininDirs = [basepath]

timeBetweenSamples = 100

Network = n.NeuralNet(0.001)

inputSpikes = n.getTrainingData(traininDirs, filter="concat15", timeBetweenSamples=timeBetweenSamples)

Network.addInputLayer(len(inputSpikes),inputSpikes)
Network.addBasicLayer(40)
Network.connectWithSTDP(0,1)
Network.connect(1,1,type='inhibitory')
#TODO find efficient way of figuring out how long the simulation has to be run for

list = [t for neuron in inputSpikes for t in neuron]
list.sort()

runTime = int(list[len(inputSpikes) - 1]) + 10
Network.run(runTime)

trainedWeights = Network.getWeights(0)
print "trainedWeights: " + str(trainedWeights)

