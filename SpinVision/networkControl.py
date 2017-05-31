import neuralNet as n
import AEDAT_Handler as f

TIME_BETWEEN_ITERATIONS = 100
ITERATIONS = 5


def trainFromFile(inputLayerSize, outputLayerSize, iterations, timeBetweenSamples, source1, source2, plot=False):
    
    net = n.NeuralNet()

    networkData = net.setUpInitialTraining(inputLayerSize, outpuLayerSize=outputLayerSize, source1=source1,
                                           source2=source2, timeBetweenSamples=timeBetweenSamples,iterations=iterations)

    runTime = int(networkData['lastSpikeAt'] + 10)
    print str(runTime) + " this is the RunTime"
    net.run(runTime / 5, record=True)

    outputPop = net.layers[networkData['outputLayer']].pop

    weights = net.connections[networkData['STDPConnection']].proj.getWeights(format="array")

    out = networkData['outputLayer']
    if plot:
        net.plotSpikes(out, block=True)

    return {'outPutLayer': out,
            'trainedWeights': weights}

def trainWithWeights(weights, iterations, tbs, plot=False):
    return