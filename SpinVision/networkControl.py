import neuralNet as n
import AEDAT_Handler as f
import matplotlib.pyplot as plt
import traceback
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import pyNN.spiNNaker as p

TIME_BETWEEN_ITERATIONS = 200
ITERATIONS = 5
RUNTIME_CONSTANT = 1
MAX_ITERATIONS_FOR_TRAINING_RUN = 20  # Todo tune this number
WMAX = 1
WMIN = 0
MEAN = 0.5
STD = 0.15


# This function is an overarhching training method, calling the sub training methods multiple times
# It is necessary, as SpiNNaker does not have enough memory to load a very long training file
# so shorter versions of the training file are loaded onto the machine again and again, to achieve high iteration training
def train(inputSize, outputSize, iterations, source1, source2, weightSource=None, weightDistr='uniform', save=False, destFile=None,
          plot=False):
    weights = None
    untrainedWeights = None

    # If we want to train from scratch (e.g. no good initial weights we can use)
    if weightSource is None:
        if weightDistr == 'uniform':
            weights = n.createUniformWeights(inputSize,outputSize, WMAX, WMIN)
        if weightDistr == 'gaussian':
            weights = n.createGaussianWeights(inputSize, outputSize, mean=MEAN, stdev=STD)

    else:
        weights = loadWeights(weightSource)

    untrainedWeights = weights

    print weights
    weights = doTrainingIterations(iterations, source1, source2, weights)

    if save:
        saveWeights(weights, destFile)

    if plot and untrainedWeights is not None:
        print "yoo plottin"
        #plot weights
        plot2DWeightsOrdered(untrainedWeights, weights, plot)
        #plot spikes
        getNetworkResponses(untrainedWeights, weights, source1, source1, plot)

    return weights


def doTrainingIterations(iterations, source1, source2, weights):

    nrLargeIterations = math.ceil(iterations / MAX_ITERATIONS_FOR_TRAINING_RUN)

    counter = 0
    while iterations > 0:
        counter += 1
        print "\n\n\n\n"
        print "///////////////////////////////////////////////////////////////////////////////////"
        print "\n\n"
        print "Training iteration: " + str(counter) + "/" + str(int(nrLargeIterations))
        print "\n\n"
        print "///////////////////////////////////////////////////////////////////////////////////"
        print "\n\n\n\n"
        currentIter = MAX_ITERATIONS_FOR_TRAINING_RUN

        if iterations - currentIter <= 0:
            currentIter = iterations

        netWorkData = trainWithWeights(weights, currentIter, source1, source2)
        iterations -= currentIter
        weights = netWorkData['trainedWeights']

    return weights


def trainFromFile(inputLayerSize, outputLayerSize, iterations, timeBetweenSamples, source1, source2, plot=False):
    #with n.NeuralNet() as net:
    net = n.NeuralNet()
    networkData = net.setUpForTraining(inputLayerSize, outpuLayerSize=outputLayerSize, source1=source1,
                                       source2=source2, timeBetweenSamples=timeBetweenSamples,
                                       iterations=iterations)

    runTime = int(networkData['lastSpikeAt'] + 10) / RUNTIME_CONSTANT
    print "runtime: " + str(runTime)
    net.run(runTime, record=True)

    weights = net.connections[networkData['STDPConnection']].proj.getWeights(format="array")

    out = networkData['outputLayer']
    if plot:
        net.plotSpikes(out, block=True)
    net.__exit__()


    return {'outPutLayer': out,
            'trainedWeights': weights}


def trainWithWeights(weights, iterations, source1, source2, plot=False):
    out = None

    net = n.NeuralNet()
    inputLayerSize = len(weights)
    outPutLayerSize = len(weights[0])  # any element of weight is fine

    networkData = net.setUpForTraining(inputLayerSize, outPutLayerSize,
                                       source1=source1, source2=source2,
                                       timeBetweenSamples=TIME_BETWEEN_ITERATIONS, iterations=iterations,
                                       weights=weights)

    runTime = int(networkData['lastSpikeAt'] + 10) / RUNTIME_CONSTANT

    net.run(runTime, record=True)

    weights = net.connections[networkData['STDPConnection']].proj.getWeights(format="array")
    out = networkData['outputLayer']

    if plot:
        net.plotSpikes(out, block=True)

    p.end()
    return {'outPutLayer': out,
            'trainedWeights': weights}


def evaluate(stimulusSource1, stimulusSource2, unWeightSource, tWeightSource, plotResponse=False):
    untrainedWeigts = loadWeights(unWeightSource)
    trainedWeights = loadWeights(tWeightSource)

    spikes = getNetworkResponses(untrainedWeigts, trainedWeights, stimulusSource1, stimulusSource2)

    untrainedSpikes = spikes['untrained']
    trainedSpikes = spikes['trained']


# This function creates an untrained and a trained network, gets their responses to stimuli
# located in source1 and source 2
def getNetworkResponses(untrainedWeights, trainedWeights, source1, source2, plot=False):
    untrainedSpikes = []
    trainedSpikes = []
    # with n.NeuralNet() as untrainedNet:
    untrainedNet = n.NeuralNet()
    # set up network
    untrainedData = untrainedNet.setUpEvaluation(untrainedWeights, source1, source2)
    # evaluate
    out = untrainedData['outputLayer']
    runTime = untrainedData['runTime'] / RUNTIME_CONSTANT
    untrainedNet.run(runTime)

    outLayer = untrainedNet.layers[out].pop
    untrainedSpikes = outLayer.getSpikes(compatible_output=True)
    p.end()
    # with n.NeuralNet() as trainedNet:
    trainedNet = n.NeuralNet()
    trainedData = trainedNet.setUpEvaluation(trainedWeights, source1, source2)
    out = trainedData['outputLayer']
    runTime = trainedData['runTime'] / RUNTIME_CONSTANT
    trainedNet.run(runTime)

    outLayer = trainedNet.layers[out].pop
    trainedSpikes = outLayer.getSpikes(compatible_output=True)

    if plot:
        plotSpikes(untrainedSpikes, trainedSpikes)

    p.end()
    return {'untrained': untrainedSpikes, 'trained': trainedSpikes}


def saveWeights(weights, destFile):
    with open(destFile, 'w') as writeTo:
        for ns in weights:
            for nd in ns:
                writeTo.write(str(nd) + ' ')
            writeTo.write('\n')


def loadWeights(sourceFile):
    weights = []
    with open(sourceFile) as readFrom:
        lines = readFrom.readlines()
        weights = [[] for line in lines]
        for i in range(len(lines)):
            weights[i] = [float(w) for w in lines[i].split()]

    return weights

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#                               PLOTTING
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def plotSpikes(untrainedSpikes, trainedSpikes, block=True):
    b, axarr = plt.subplots(2, sharex=True, sharey=True)
    axarr[0].plot(untrainedSpikes[:, 1], untrainedSpikes[:, 0], ls='', marker='|', markersize=8, ms=1)
    axarr[0].set_title('Before Training')
    axarr[1].set_title('After Training')
    axarr[1].plot(trainedSpikes[:, 1], trainedSpikes[:, 0], ls='', marker='|', markersize=8, ms=1)

    axarr[0].grid()
    axarr[1].grid()
    plt.show(block=block)

def plot2DWeightsOrdered(untrainedWeights, trainedWeights, block=True):
    nrBuckets = 100 # as weights are between 0 and 1 this should bask them into a 100
                    # buckets
    barWidht = 0.01
    untrainedBuckets = [0] * (nrBuckets + 1)
    untrainedWeights = [int(w * 100) for neuron in untrainedWeights for w in neuron]

    nrWeights = len(untrainedWeights)

    for weight in untrainedWeights:
        assert weight >= 0
        if weight > 100:
            weight = 100 #TODO do something with this
        assert weight <= 100

        untrainedBuckets[weight] += 1


    trainedBuckets = [0] * (nrBuckets + 1)
    trainedWeights = [int(w * 100) for neuron in trainedWeights for w in neuron]

    for weight in trainedWeights:
        assert weight >= 0
        if weight > 100:
            weight = 100  # TODO do something with this
        assert weight <= 100
        trainedBuckets[weight] += 1

    scale = [0.01 * i for i in range(nrBuckets + 1)]

    assert len(scale) == len(untrainedBuckets)
    assert len(scale) == len(trainedBuckets)

    sumBucket = [0,0]
    for n in untrainedBuckets:
        sumBucket[0] += n
    for n in trainedBuckets:
        sumBucket[1] += n
    print "weights in buckst vs nrweights"
    print sumBucket
    print nrWeights


    untrainedBuckets = [float(float(n) / nrWeights) for n in untrainedBuckets]
    trainedBuckets = [float(float(n) / nrWeights) for n in trainedBuckets]

    f, axarr = plt.subplots(2, sharex=True, sharey=True)
    bar1 = axarr[0].bar(scale, untrainedBuckets, barWidht, color='r')
    axarr[0].set_title('Before Training')
    axarr[1].set_title('After Training')
    bar2 =  axarr[1].bar(scale, trainedBuckets, barWidht, color='b')

    plt.show(block=block)

