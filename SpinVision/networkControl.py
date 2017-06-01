import neuralNet as n
import AEDAT_Handler as f
import matplotlib.pyplot as plt

TIME_BETWEEN_ITERATIONS = 200
ITERATIONS = 5
RUNTIME_CONSTANT = 1
MAX_ITERATIONS_FOR_TRAINING_RUN = 40 #Todo tune this number

# This function is an overarhching training method, calling the sub training methods multiple times
# It is necessary, as SpiNNaker does not have enough memory to load a very long training file
# so shorter versions of the training file are loaded onto the machine again and again, to achieve high iteration training
def train(inputSize, outputSize, iterations, source1, source2, save=False, destFile=None, plot=False):
    netWorkData = trainFromFile(inputSize, outputSize, 1
                            , TIME_BETWEEN_ITERATIONS, source1, source2)
    weights = netWorkData['trainedWeights']
    iterations -= 1

    plotSpikes = False
    print iterations
    while iterations > 0:
        currentIter = MAX_ITERATIONS_FOR_TRAINING_RUN

        if iterations - currentIter <= 0:
            currentIter = iterations
            if plot:
                plotSpikes = True

        netWorkData = trainWithWeights(weights, currentIter,
                                       source1, source2, plotSpikes)

        weights = netWorkData['trainedWeights']

        iterations -= currentIter




def trainFromFile(inputLayerSize, outputLayerSize, iterations, timeBetweenSamples, source1, source2, plot=False):
    with n.NeuralNet() as net:
        networkData = net.setUpForTraining(inputLayerSize, outpuLayerSize=outputLayerSize, source1=source1,
                                           source2=source2, timeBetweenSamples=timeBetweenSamples,
                                           iterations=iterations)

        runTime = int(networkData['lastSpikeAt'] + 10) / RUNTIME_CONSTANT
        net.run(runTime, record=True)

        weights = net.connections[networkData['STDPConnection']].proj.getWeights(format="array")

        out = networkData['outputLayer']
        if plot:
            net.plotSpikes(out, block=True)

        return {'outPutLayer': out,
                'trainedWeights': weights}


def trainWithWeights(weights, iterations, source1, source2, plot=False):
    out = None
    with n.NeuralNet() as net:

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
    with n.NeuralNet() as untrainedNet:
        # set up network
        untrainedData = untrainedNet.setUpEvaluation(untrainedWeights, source1, source2)
        # evaluate
        out = untrainedData['outputLayer']
        runTime = untrainedData['runTime'] / RUNTIME_CONSTANT
        untrainedNet.run(runTime)

        outLayer = untrainedNet.layers[out].pop
        untrainedSpikes = outLayer.getSpikes(compatible_output=True)

    with n.NeuralNet() as trainedNet:

        trainedData = trainedNet.setUpEvaluation(trainedWeights, source1, source2)
        out = trainedData['outputLayer']
        runTime = trainedData['runTime'] / RUNTIME_CONSTANT
        trainedNet.run(runTime)

        outLayer = trainedNet.layers[out].pop
        trainedSpikes = outLayer.getSpikes(compatible_output=True)

    if plot:
        plotEval(untrainedSpikes, trainedSpikes)

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

def plotEval(untrainedSpikes, trainedSpikes):
    b, axarr = plt.subplots(2, sharex=True, sharey=True)
    axarr[0].plot(untrainedSpikes[:, 1], untrainedSpikes[:, 0], ls='', marker='|', markersize=8, ms=1)
    axarr[0].set_title('Before Training')
    axarr[1].set_title('After Training')
    axarr[1].plot(trainedSpikes[:, 1], trainedSpikes[:, 0], ls='', marker='|', markersize=8, ms=1)

    axarr[0].grid()
    axarr[1].grid()
    plt.show()