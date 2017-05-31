import neuralNet as n
import AEDAT_Handler as f

TIME_BETWEEN_ITERATIONS = 10
ITERATIONS = 5
RUNTIME_CONSTANT = 5
MAX_ITERATIONS_FOR_TRAINING_RUN = 10 #Todo tune this number


def train(inputSize, outputSize, iterations, source1, source2, save=False, destFile=None, plot=False):
    netWorkData = trainFromFile(inputSize, outputSize, MAX_ITERATIONS_FOR_TRAINING_RUN
                            , TIME_BETWEEN_ITERATIONS, source1, source2)
    weights = netWorkData['trainedWeights']
    iterations -= MAX_ITERATIONS_FOR_TRAINING_RUN

    plotSpikes = False
    print iterations
    while iterations > 0:
        if iterations - MAX_ITERATIONS_FOR_TRAINING_RUN <= 0 \
                and plot:
            plotSpikes = True

        netWorkData = trainWithWeights(weights, MAX_ITERATIONS_FOR_TRAINING_RUN,
                                       source1, source2, plotSpikes)

        weights = netWorkData['trainedWeights']

        iterations -= MAX_ITERATIONS_FOR_TRAINING_RUN


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
        #print 'yoo'
        inputLayerSize = len(weights)
        #print weights
        outPutLayerSize = len(weights[0])  # any element of weight is fine
        networkData = net.setUpForTraining(inputLayerSize, outPutLayerSize,
                                           source1=source1, source2=source2,
                                           timeBetweenSamples=TIME_BETWEEN_ITERATIONS, iterations=iterations,
                                           weights=weights)

        runTime = int(networkData['lastSpikeAt'] + 10) / RUNTIME_CONSTANT

        net.run(runTime, record=True)
       # print 'raaaan'

        weights = net.connections[networkData['STDPConnection']].proj.getWeights(format="array")
        out = networkData['outputLayer']

        if plot:
            net.plotSpikes(out, block=True)

        print "about to return"

    return {'outPutLayer': out,
            'trainedWeights': weights}


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