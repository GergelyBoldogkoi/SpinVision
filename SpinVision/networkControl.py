import neuralNet as n
import AEDAT_Handler as f
import matplotlib.pyplot as plt
import traceback
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

np.set_printoptions(suppress=True)
import math
import pyNN.spiNNaker as p
import plotly.plotly as py
import plotly.graph_objs as go

TIME_BETWEEN_ITERATIONS = 500
ITERATIONS = 5
RUNTIME_CONSTANT = 1
MAX_ITERATIONS_FOR_TRAINING_RUN = 25  # Todo tune this number
WMAX = 0.1  # n.__STDPParameters__['wMax']
WMIN = n.__STDPParameters__['wMin']
MEAN = n.__STDPParameters__['mean']
STD = n.__STDPParameters__['std']


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#                              Train End-Positions
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def train_endPositions(nrEndPositions, sources, trainedWeights):

    assert nrEndPositions == len(sources)
    # Find out which neurons respond to which input
    pairings = pairInputsWithNeurons(sources, trainedWeights)

    nrInputNeurons = len(trainedWeights)
    nrTrajectoryNeurons = len(trainedWeights[0])


    net = n.NeuralNet()

    #create trajectory and end-position layer
    trajectoryLayer = net.addLayer(nrTrajectoryNeurons, n.__neuronType__, n.__neuronParameters__)
    positionLayer = net.addLayer(nrEndPositions, n.__neuronType__, n.__neuronParameters__)
    traj_posConnection = connectTrajectoryAndPositionLayer(net, pairings, positionLayer, trajectoryLayer)

    return net

def evaluateEndPositions(nrInputNeurons, sources, inputWeights, trainedNetwork):

    evalData = n.getTrainingData(nrInputNeurons, sourceFiles=sources, iterations=1, timebetweenSamples=500)
    evalSpikes = evalData['spikeTimes']
    lastSpike = evalData['lastSpikeAt']


    inputLayer = trainedNetwork.addInputLayer(nrInputNeurons, evalSpikes)

    connections = n.createConnectionsFromWeights(inputWeights)

    #as the trajectory layer is added first to network its number is 9 #TODO get rid of this magic number
    input_trajConnection = trainedNetwork.connect(inputLayer, 0, p.FromListConnector(connections))

    #todo do something with the fact that now all output positions are going to spike if the connected neuron spikes (which it does)
    runTime = int(lastSpike + 100) / RUNTIME_CONSTANT
    trainedNetwork.run(runTime=runTime, record=True)

    trajSpikes = trainedNetwork.layers[0].pop.getSpikes(compatible_output=True)
    print "traj Spikes"
    print trajSpikes

    spikes = trainedNetwork.layers[1].pop.getSpikes(compatible_output=True)
    print "spikes"
    print spikes

    p.end() #disconnect from SpiNNaker

    plotSpikes([], trajSpikes, block=True)
    plotSpikes([], spikes, block=True)

    return spikes

def connectTrajectoryAndPositionLayer(net, pairings, positionLayer, trajectoryLayer):
    connections = []
    nrPosNeuron = 0
    for sourceNeuron in pairings.values():  # Allocate all learned trajectories to an end-position neuron
        connections.append((sourceNeuron, nrPosNeuron, 20, n.__STDPParameters__[                                   'delay']))  # TODO improve this when multiple trajectories represent the same end pos
        nrPosNeuron += 1
    traj_posConnection = net.connect(trajectoryLayer, positionLayer,
                                     connectivity=p.FromListConnector(connections))  # type='excitatory' by default
    return traj_posConnection

def pairInputsWithNeurons(sources, trainedWeights):
    pairings = {}
    for recordingName in sources:
        formattedSpikes = []
        for i in range(len(trainedWeights[0])): # For each trajectory neuron
            formattedSpikes.append([])

        spikes = get2LayerNetworkResponses(None, trainedWeights, [recordingName], plot=False)['trained']

        for spike in spikes: # reformat spikes, so that we get rid of the numpy datatypes
            formattedSpikes[int(spike[0])].append(spike[1])

        neuronId = findMostSpikingNeuron(formattedSpikes)  # find neuron that got most excited
        pairings[recordingName] = neuronId

    return pairings


def findMostSpikingNeuron(spikes):
    currmax = 0
    neuron = None
    for neuronID in range(len(spikes)):
        if len(spikes[neuronID]) > currmax:
            currmax = len(spikes[neuronID])
            neuron = neuronID
    return neuron

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#                              Train Trajectories
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# This function is an overarhching training method, calling the sub training methods multiple times
# It is necessary, as SpiNNaker does not have enough memory to load a very long training file
# so shorter versions of the training file are loaded onto the machine again and again, to achieve high iteration training



def train_Trajectories(inputSize, outputSize, iterations, sources, weightSource=None, weightDistr='uniform', save=False,
                       destFile=None,
                       plot=False, weights=None):
    untrainedWeights = None

    # If we want to train from scratch (e.g. no good initial weights we can use)
    if weightSource is None:
        if weightDistr == 'uniform':
            weights = n.createUniformWeights(inputSize, outputSize, WMAX, WMIN)
        if weightDistr == 'gaussian':
            weights = n.createGaussianWeights(inputSize, outputSize, mean=MEAN, stdev=STD)

    else:
        weights = loadWeights(weightSource)

    print "Trained first trajectory"
    untrainedWeights = weights

    weights = do_TrainingIterations(iterations, [sources[0]], weights)

    firstWeights = weights

    weights = do_TrainingIterations(iterations, [sources[1]], weights)

    displayInfo(plot, [sources[0]], untrainedWeights, firstWeights)

    print "Trained second trajectory"
    displayInfo(plot, [sources[0], sources[1]], untrainedWeights, weights)
    if save:
        saveWeights(weights, destFile)

    return weights


def do_TrainingIterations(iterations, sources, weights, startFromNeuron=0):
    nrLargeIterations = math.ceil(float(iterations) / float(MAX_ITERATIONS_FOR_TRAINING_RUN))

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

        netWorkData = train_TrajectoriesWithWeights(weights, currentIter, sources)
        iterations -= currentIter
        weights = netWorkData['trainedWeights']
        # print " weights in iteration " + str(counter)
        # printWeights(weights)

    return weights


def train_TrajectoriesFromFile(inputLayerSize, outputLayerSize, iterations, timeBetweenSamples, sources, plot=False,
                               startFromNeuron=0):
    # with n.NeuralNet() as net:
    net = n.NeuralNet()
    networkData = net.setUp2LayersForTraining(inputLayerSize, outpuLayerSize=outputLayerSize, sources=sources,
                                              timeBetweenSamples=timeBetweenSamples,
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


def train_TrajectoriesWithWeights(weights, iterations, sources, plot=False, startFromNeuron=0):
    out = None

    net = n.NeuralNet()
    inputLayerSize = len(weights)
    outPutLayerSize = len(weights[0])  # any element of weight is fine

    networkData = net.setUp2LayersForTraining(inputLayerSize, outPutLayerSize,
                                              sources=sources,
                                              timeBetweenSamples=TIME_BETWEEN_ITERATIONS, iterations=iterations,
                                              weights=weights)

    runTime = int(networkData['lastSpikeAt'] + 10) / RUNTIME_CONSTANT
    # print "weights before trainin"
    # printWeights(weights)
    net.run(runTime, record=True)

    weights = net.connections[networkData['STDPConnection']].proj.getWeights(format="array")
    # print "weights aftergettin em from connections"
    # printWeights(weights)
    out = networkData['outputLayer']

    if plot:
        net.plotSpikes(out, block=True)

    p.end()

    return {'outPutLayer': out,
            'trainedWeights': weights}

# This function creates an untrained and a trained network, gets their responses to stimuli
# located in source1 and source 2
def get2LayerNetworkResponses(untrainedWeights, trainedWeights, sources, plot=False, startFromNeuron=0):
    untrainedSpikes = []
    trainedSpikes = []
    if untrainedWeights is not None:
        # with n.NeuralNet() as untrainedNet:
        untrainedNet = n.NeuralNet()
        # set up network
        untrainedData = untrainedNet.setUp2LayerEvaluation(untrainedWeights, sources, startFromNeuron=startFromNeuron)
        # evaluate
        out = untrainedData['outputLayer']
        runTime = untrainedData['runTime'] / RUNTIME_CONSTANT
        untrainedNet.run(runTime)

        outLayer = untrainedNet.layers[out].pop
        untrainedSpikes = outLayer.getSpikes(compatible_output=True)
        p.end()
    # with n.NeuralNet() as trainedNet:
    if trainedWeights is not None:
        trainedNet = n.NeuralNet()
        trainedData = trainedNet.setUp2LayerEvaluation(trainedWeights, sources, startFromNeuron=startFromNeuron)
        out = trainedData['outputLayer']
        runTime = trainedData['runTime'] / RUNTIME_CONSTANT
        trainedNet.run(runTime)

        outLayer = trainedNet.layers[out].pop
        trainedSpikes = outLayer.getSpikes(compatible_output=True)

    if plot:
        if trainedWeights is not None:
            if untrainedWeights is not None:
                plotSpikes(untrainedSpikes, trainedSpikes)
            else:
                plotSpikes([], trainedSpikes)
        elif untrainedWeights is not None:
            plotSpikes(untrainedWeights, [])

    p.end()
    return {'untrained': untrainedSpikes, 'trained': trainedSpikes}




# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#                               PLOTTING
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def plotSpikes(untrainedSpikes, trainedSpikes, block=True):
    b, axarr = plt.subplots(2, sharex=True, sharey=True)
    if len(untrainedSpikes) != 0:
        axarr[0].plot(untrainedSpikes[:, 1], untrainedSpikes[:, 0], ls='', marker='|', markersize=8, ms=1)
        axarr[0].set_title('Before Training')
        axarr[0].grid()

    if len(trainedSpikes) != 0:
        axarr[1].set_title('After Training')
        axarr[1].plot(trainedSpikes[:, 1], trainedSpikes[:, 0], ls='', marker='|', markersize=8, ms=1)
        axarr[1].grid()
    plt.show(block=block)


def plot2DWeightsOrdered(untrainedWeights, trainedWeights, block=True):
    nrBuckets = 100  # as weights are between 0 and 1 this should bask them into a 100
    # buckets
    barWidht = 0.01
    untrainedBuckets = [0] * (nrBuckets + 1)
    untrainedWeights = [int(w * 100) for neuron in untrainedWeights for w in neuron]

    nrWeights = len(untrainedWeights)

    for weight in untrainedWeights:
        assert weight >= 0
        if weight > 100:
            weight = 100  # TODO do something with this
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

    sumBucket = [0, 0]
    for n in untrainedBuckets:
        sumBucket[0] += n
    for n in trainedBuckets:
        sumBucket[1] += n

    untrainedBuckets = [float(float(n) / nrWeights) for n in untrainedBuckets]
    trainedBuckets = [float(float(n) / nrWeights) for n in trainedBuckets]

    f, axarr = plt.subplots(2, sharex=True, sharey=True)
    bar1 = axarr[0].bar(scale, untrainedBuckets, barWidht, color='r')
    axarr[0].set_title('Before Training')
    axarr[1].set_title('After Training')
    bar2 = axarr[1].bar(scale, trainedBuckets, barWidht, color='b')

    plt.show(block=block)


def plotWeightHeatmap(untrainedWeights, trainedWeights, block=False):
    f, axarr = plt.subplots(2, sharex=True, sharey=True)
    for lis in trainedWeights:
        for weight in lis:
            if weight > 1:
                weight = 1
    # print "weights after maxing them"
    # print weights
    # print "weights after training"
    # print trainedWeights

    axarr[0].imshow(untrainedWeights, cmap='hot', interpolation='nearest')
    axarr[1].imshow(trainedWeights, cmap='hot', interpolation='nearest')

    axarr[0].set_title('Before Training')
    axarr[1].set_title('After Training')

    plt.show(block=block)


def printWeights(weights):
    for i in range(len(weights)):
        print "Source: " + str(i)
        print weights[i]


def displayInfo(plot, sources, untrainedWeights, weights, startFromNeuron=0):
    avgChange = getAvgChangeInWeights(untrainedWeights, weights)
    print " average change in weights: " + str(avgChange)
    print "that is a change of " + str(getAvgChangeInWeights(untrainedWeights, weights) * 100) + '%'
    weightDec = getDecreaseInWeights(untrainedWeights, weights)
    print "Average decrease in weight: " + str(weightDec['avg'])
    print "that is a decrease of " + str(weightDec['avg'] * 100) + '%'
    print "Number of synapses that have decreased in weight: " + str(weightDec['nr'])
    print "That means the percentage of synapses that decreased in weight is: " \
          + str(float((float(weightDec['nr']) / (len(untrainedWeights) * len(untrainedWeights[0]))) * 100)) + "%"
    # print "UNTrained weights"
    # print untrainedWeights
    # print "Trained weights"
    # print weights
    maxweight = max(w for neuron in weights for w in neuron)
    print "Max weight: " + str(maxweight)
    if plot and untrainedWeights is not None:
        # plot weights
        plot2DWeightsOrdered(untrainedWeights, weights, plot)
        plotWeightHeatmap(untrainedWeights, weights, plot)
        # plot spikes
        get2LayerNetworkResponses(untrainedWeights, weights, sources, plot, startFromNeuron=startFromNeuron)

def getAvgChangeInWeights(untrainedWeights, trainedWeights):
    change = 0
    for i in range(len(untrainedWeights)):
        for j in range(len(untrainedWeights[0])):
            change += abs(untrainedWeights[i][j] - trainedWeights[i][j])
    change = float(float(change) / (len(untrainedWeights) * len(untrainedWeights[0])))
    return change

def getDecreaseInWeights(untrainedWeights, trainedWeights):
    nrOfDecreases = 0
    avgDecrease = 0
    for i in range(len(untrainedWeights)):
        for j in range(len(untrainedWeights[0])):
            if untrainedWeights[i][j] > trainedWeights[i][j]:
                avgDecrease += untrainedWeights[i][j] - trainedWeights[i][j]
                nrOfDecreases += 1
    avgDecrease = float(float(avgDecrease) / (len(untrainedWeights) * len(untrainedWeights[0])))
    return {'avg': avgDecrease, 'nr': nrOfDecreases}


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#                               Helper Functions
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# this function is used to count the number of neurons activated by a certain recording.
# It will be used in taining to ensure that for the recordings after the first, not the same neurons are trained.
# WARNING! if source is given, data will be ignored
def countNeurons(source=None, data=None):
    if source is not None:
        data = f.readData(source)

    neurons = []
    for i in range(len(data.x)):
        x = data.x[i]
        y = data.y[i]
        if (x, y) not in neurons:
            neurons.append((x, y))

    return len(neurons)



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
