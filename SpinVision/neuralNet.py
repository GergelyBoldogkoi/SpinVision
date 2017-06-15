import pyNN.spiNNaker as p
import matplotlib.pyplot as plt
import pylab
from collections import namedtuple
import AEDAT_Handler as f
import random as r
import numpy as np
import math

INHIB_WEIGHT = 5
Layer = namedtuple("Layer", "pop nType nParams")
Connection = namedtuple("Connection", "proj pre post connectivity type")


# inputFormat: layerDescriptors = [[pop L1, Neuron Type L1, Neuron Params L1],[Pop L2,...]...]
__neuronParameters__ = {
    'cm': 12,  # The capacitance of the LIF neuron in nano-Farads
    'tau_m': 110,  # The time-constant of the RC circuit, in millisecon
    'tau_refrac': 40.0,  # The refractory period, in milliseconds
    'v_reset': -70.0,  # The voltage to set the neuron at immediately after a spike
    'v_rest': -65,  # The ambient rest voltage of the neuron
    'v_thresh': -61,  # The threshold voltage at which the neuron will spike
    'tau_syn_E': 2.0,  # The excitatory input current decay time-constant
    'tau_syn_I': 25.0,  # The inhibitory input current decay time-constant
    'i_offset': 0.0  # A base input current to add each timestep
}
__neuronType__ = p.IF_curr_exp

__STDPParameters__ = {
    'mean': 0.5,
    'std': 0.75,
    'delay': 1,
    'weightRule': 'multiplicative',
    'tauPlus': 15,
    'tauMinus': 30,
    'wMax': 1,
    'wMin': 0,
    'aPlus': 0.1,
    'aMinus': 0.1,
    'weightInit': 'uniform'
}


class NeuralNet(object):
    # __neuronType__ = p.IF_curr_exp
    # __neuronParameters__ = {
    #     'cm': 12,  # The capacitance of the LIF neuron in nano-Farads
    #     'tau_m': 110,  # The time-constant of the RC circuit, in milliseconds
    #     'tau_refrac': 40,  # The refractory period, in milliseconds
    #     'v_reset': -70.0,  # The voltage to set the neuron at immediately after a spike
    #     'v_rest': -65,  # The ambient rest voltage of the neuron
    #     'v_thresh': -61,  # The threshold voltage at which the neuron will spike
    #     'tau_syn_E': 5.0,  # The excitatory input current decay time-constant
    #     'tau_syn_I': 10,  # The inhibitory input current decay time-constant
    #     'i_offset': 0.0  # A base input current to add each timestep
    # }
    # __STDPParameters__ = {
    #     'mean': 0.5,
    #     'std': 0.15,
    #     'delay': 1,
    #     'weightRule': 'additive',
    #     'tauPlus': 50,
    #     'tauMinus': 60,
    #     'wMax': 1,
    #     'wMin': 0,
    #     'aPlus': 0.05,
    #     'aMinus': 0.05,
    #     'weightInit': 'uniform'
    # }


    def __init__(self, timeStep=1):
        p.setup(timestep=timeStep)
        self.layers = []  # This list should contain Layers, see namedTuple def
        self.connections = []  # This list should contain Connections, see namedtuple def
        self.runTime = None
        self.sampleTimes = None
        self.annotations = []

    def __enter__(self):
        p.setup(timestep=1)
        self.layers = []  # This list should contain Layers, see namedTuple def
        self.connections = []  # This list should contain Connections, see namedtuple def
        self.runTime = None
        self.sampleTimes = None
        self.annotations = []
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print exc_type, exc_value, traceback
        self.Layers = None
        self.connections = None
        self.runTime = None
        self.sampleTimes = None
        self.annotations = None
        p.end()
        return self

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                           LAYER MANIPULATION
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # Adds a default population of neurons to the network
    def addBasicLayer(self, size):
        pop = p.Population(size, p.IF_curr_exp, {})
        layer = Layer(pop, p.IF_curr_exp, {})

        self.layers.append(layer)
        return len(self.layers) - 1

    # adds layer to network
    def addLayer(self, size, neuronType, neuronParams):
        pop = p.Population(size, neuronType, neuronParams)
        layer = Layer(pop, neuronType, neuronParams)
        self.layers.append(layer)

        return len(self.layers) - 1

    # Adds a population with pre-determined spike-times to the Network
    def addInputLayer(self, size, sourceSpikes):
        spikes = []
        for i in range(0, len(sourceSpikes)):
            spikes.append(sourceSpikes[i])

        pop = p.Population(size, p.SpikeSourceArray, {'spike_times': spikes})
        layer = Layer(pop, p.SpikeSourceArray, {'spike_times': spikes})
        self.layers.append(layer)

        return len(self.layers) - 1

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                           CONNECTION MANIPULATION
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # Connects two layers of the network
    def connect(self, preSynaptic, postSynaptic,
                connectivity=p.AllToAllConnector(weights=5, delays=1),
                type='excitatory'):

        proj = p.Projection(self.layers[preSynaptic].pop, self.layers[postSynaptic].pop, connectivity, target=type)
        connection = Connection(proj, preSynaptic, postSynaptic, connectivity, type)

        self.connections.append(connection)
        return len(self.connections) - 1

    def connectWithSTDP(self, pre, post, initWeightMean=0.5, initWeightStd=0.15, delay=1,
                        weightMod='additive',
                        tauPlus=20, tauMinus=20,
                        wMax=1, wMin=0,
                        aPlus=0.05, aMinus=0.05, weights=None, initWeightDistr='gaussian'):

        # Maybe extend to save parameters of STDP Connection
        timingRule = p.SpikePairRule(tau_plus=tauPlus, tau_minus=tauMinus)

        # -------- Setting Weight Rule ------------
        weightRule = None
        if weightMod == 'additive':
            weightRule = p.AdditiveWeightDependence(w_max=wMax, w_min=wMin, A_plus=aPlus, A_minus=aMinus)
        elif weightMod == 'multiplicative':
            weightRule = p.MultiplicativeWeightDependence(w_min=wMin, w_max=wMax, A_plus=aPlus, A_minus=aMinus)
        else:
            raise TypeError(str(weightMod) + " is not a known weight modification rule for STDP \n"
                                             "try \'additive\' or \'multiplicative\'")
        # ------------------------------------------

        STDP_Model = p.STDPMechanism(timing_dependence=timingRule, weight_dependence=weightRule)

        # SET UP WEIGHTS OF TH ENETWORK
        #weights enter fine up to here

        if weights is None:  # if there are no weighs supplied create randomised weights
            if initWeightDistr == 'gaussian':
                connections = createGaussianConnections(self.layers[pre][0].size, self.layers[post][0].size,
                                                        initWeightMean, initWeightStd, delay)
            elif initWeightDistr == 'uniform':
                connections = createUniformConnections(self.layers[pre][0].size, self.layers[post][0].size,
                                                       wMax, wMin, delay)
            else:
                print "Unknown distribution pattern for initial weights of STDP connections: " + str(initWeightDistr)
        else:  # if weights are supplied set up STDP with those
            connections = createConnectionsFromWeights(weights)

        connectivity = p.FromListConnector(connections)

        proj = p.Projection(self.layers[pre].pop, self.layers[post].pop, connectivity,
                            synapse_dynamics=p.SynapseDynamics(slow=STDP_Model))

        connection = Connection(proj, pre, post, connectivity, 'STDP')
        self.connections.append(connection)

        return len(self.connections) - 1

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                           OPERATION
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # runs simulation
    def run(self, runTime, record=True, record_v=False):
        self.runTime = runTime

        for layer in self.layers:
            if record:
                layer.pop.record()
            if record_v:
                if 'spike_times' not in layer.nParams:  # It is not possible to record voltages for inputLayers
                    layer.pop.record_v()

        p.run(runTime)

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                           SETUPS
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    # sets up 2 layers for training
    # when this function save, it is not going to save the actual training data, but the one that has been
    # iterated iterations times
    def setUp2LayersForTraining(self, inputLayerSize, outpuLayerSize, sources, timeBetweenSamples, iterations=1,
                                weights=None):

        if (not len(self.layers) == 0) and (not len(self.connections) == 0):
            raise TypeError(
                "Network has already been initialized, please ensure to call this function on an uninitialized network")

        bundle = getTrainingData(inputLayerSize, sources, iterations, timeBetweenSamples)
        lastSpike = bundle['lastSpikeAt']
        trainingSpikes = bundle['spikeTimes']

        inputLayerNr = self.addInputLayer(inputLayerSize, trainingSpikes)

        outputLayerNr = self.addLayer(outpuLayerSize, __neuronType__, __neuronParameters__)

        STDP_Params = __STDPParameters__
        stdpNr = self.connectWithSTDP(inputLayerNr, outputLayerNr,
                                      initWeightMean=STDP_Params['mean'], initWeightStd=STDP_Params['std'], delay= STDP_Params['delay'],
                                     weightMod= STDP_Params['weightRule'], tauPlus= STDP_Params['tauPlus'],
                                     tauMinus= STDP_Params['tauMinus'], wMax= STDP_Params['wMax'],
                                     wMin= STDP_Params['wMin'], aPlus= STDP_Params['aPlus'], aMinus= STDP_Params['aMinus'],
                                     weights= weights,initWeightDistr= STDP_Params['weightInit'])

        inhibitoryNr = self.connect(outputLayerNr, outputLayerNr,
                                    connectivity=p.AllToAllConnector(weights=INHIB_WEIGHT, delays=STDP_Params['delay']),
                                    type='inhibitory')

        return {'inputLayer': inputLayerNr, 'outputLayer': outputLayerNr,
                'inhibitoryConnection': inhibitoryNr, 'STDPConnection': stdpNr,
                'lastSpikeAt': lastSpike}

    def setUp2LayerEvaluation(self, weights, sources, delay=1, startFromNeuron=0):
        # len(weights[0]) returns how many output neurons there are,
        # as all inputs are connected to all outputs, so it doesn't really matter
        # precisely which element of weights we take
        nrOut = len(weights[0])
        nrIn = len(weights)

        outputLayerNr = self.addLayer(nrOut, __neuronType__, __neuronParameters__)

        bundle = getTrainingData(nrIn, sources, 1, 100)
        evalSpikes = bundle['spikeTimes']

        lastSpike = bundle['lastSpikeAt']

        inputLayerNr = self.addInputLayer(nrIn, evalSpikes)

        self.sampleTimes = bundle['sampleTimes']

        neuronConnections = [0] * nrIn * nrOut
        for i in range(nrIn):
            for j in range(nrOut):
                neuronConnections[i * nrOut + j] = (i, j, weights[i][j], delay)

        connection = self.connect(inputLayerNr, outputLayerNr, p.FromListConnector(neuronConnections),
                                  type='excitatory')

        return {'inputLayer': inputLayerNr, 'outputLayer': outputLayerNr,
                'connection': connection, 'runTime': lastSpike}

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                           TRAINING
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # This function trains the network







    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                           OUTPUT
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # plots spikes of a layer
    def plotSpikes(self, layerId, marker='|', block=False, delayMargin=10):
        spikes = self.layers[layerId].pop.getSpikes(compatible_output=True)
        print type(spikes)
        print spikes

        ylim = self.layers[layerId].pop.size

        plt.plot(spikes[:, 1], spikes[:, 0], ls='', marker=marker, markersize=8, ms=1)
        # for element in self.sampleTimes:
        #     print "delim at: " + str(element)
        #     plt.plot((element - delayMargin, element - delayMargin), (0, ylim), color='k')

        plt.xlim((-1, self.runTime))
        plt.ylim((-0.1, ylim))
        plt.xlabel('time (t)')
        plt.ylabel('neuron index')
        plt.grid()
        plt.show(block=block)

    def getWeights(self, connection):

        return self.connections[connection][0].getWeights()


# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#                           DATA-HANDLING
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


def readSpikes(aedata, timeBetweenIterations=0, ONOFF=False):
    organisedData = {}  # datastructure containing a structure of spiketimes for each neuron
    # a new neuron is created for each x,y and ONOFF value
    spikeTime = 0
    sampleTimes_ms = []
    sampleEnd = None

    for sample in aedata:
        sampleLength = (sample.ts[len(sample.ts) - 1] - sample.ts[0]) / 1000
        data = f.extractData(sample)

        for i in range(len(data['ts'])):
            if ONOFF:
                neuronId = (data['X'][i], data['Y'][i], data['t'][i])  # x,y,ONOFF
            else:
                neuronId = (data['X'][i], data['Y'][i])

            if neuronId not in organisedData:
                organisedData[neuronId] = [spikeTime / 1000]

            else:
                organisedData[neuronId].append(spikeTime / 1000)

            if i + 1 < len(data['ts']):
                spikeTime += data['ts'][i + 1] - data['ts'][i]  # set time of next spike

        sampleEnd = spikeTime / 1000

        sampleTimes_ms.append(sampleLength + timeBetweenIterations)

        spikeTime += timeBetweenIterations * 1000  # leave timeBetweenSamples time between samples

    return {'data': organisedData, 'sampleTimes': sampleTimes_ms, 'lastSpikeAt': sampleEnd}


# This function reads in spikes from all files in given directories
def getTrainingDataFromDirectories(trainingDirectories, filter=None, iterations=1, timeBetweenSamples=0, startFrom_ms=0,
                                   save=False,
                                   destFile=None):
    # type: ([str], str, int,  int, int, bool, str) -> [[float]]

    aedata = f.concatenate(trainingDirectories, timeBetweenSamples * 1000, filter, destFile, save)

    cont = readSpikes([aedata], timeBetweenIterations=timeBetweenSamples)
    data = cont['data']
    finishesAt = cont['lastSpikeAt']
    spikeTimes = []
    for neuronSpikes in data.values():
        neuronSpikes.sort()
        spikeTimes.append(neuronSpikes)

    return {'spikeTimes': spikeTimes, 'lastSpikeAt': finishesAt}


def getTrainingData(inputlayerSize, sourceFiles, iterations, timebetweenSamples, randomise=False):
    aedata = []
    for file in sourceFiles:
        for i in range(iterations):
            aedata.append(f.readData(file))
    # aedata2 = f.readData(sourceFile2)
    #
    # if randomise: # todo make a list of inputs be handleable
    #     ordering = orderRandomly(aedata1, aedata2, iterations)
    # else:
    #     ordering = []
    #     for i in range(iterations):
    #         ordering.append(aedata1)
    #         ordering.append(aedata2)

    ordering = aedata

    cont = readSpikes(ordering, timebetweenSamples)
    data = cont['data']

    spikeTimes = []
    # create 2d array of neuron layer (such that x,y coordinates are applicable)
    layerWidth = int(math.ceil(math.sqrt(inputlayerSize)))
    for x in range(layerWidth):
        for y in range(layerWidth):
            spikeTimes.append([])


    for neuron in data:
        x = int(neuron[0])
        if x == 128:
            x = 0
        y = int(neuron[1])
        index = (x-1) * layerWidth + (y-1)  # the -1 terms come from the fact that the input from the DVS
                                            # starts numbering the coordinates from 1

        if index > len(spikeTimes):
            print "ERROR ignoring input"    # Interesting thing: SOMETIMES neuron with x=128 gets passed in todo investigate
                                            # about 5 spikes, so should be fine
            print "x " + str(x)
            print "y " + str(y)
            print "width spiketimes " + str(layerWidth)
        else:

            spikeTimes[index] = data.get(neuron)

    # print "spikeTimes"
    # print len(spikeTimes)
    # print spikeTimes

    # for neuronSpikes in data.values():
    #     neuronSpikes.sort()
    #     spikeTimes.append(neuronSpikes)
    #
    # fillFrom = len(spikeTimes)
    # for i in range(fillFrom, inputlayerSize):
    #     spikeTimes.append([])

    return {'spikeTimes': spikeTimes,
            'sampleTimes': cont['sampleTimes'],
            'lastSpikeAt': cont['lastSpikeAt']}






    # get spikes, and length, and time in betweenem
    # win


# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#                           HELPER FUNCTIONS
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def createGaussianConnections(nrPreNeurons, nrPostNeurons, mean, stdev, delay=1):
    nrConnections = nrPreNeurons * nrPostNeurons
    connections = [0] * nrConnections

    i = 0
    for ns in range(nrPreNeurons):
        for nd in range(nrPostNeurons):
            weight = abs(r.gauss(mean, stdev))
            connections[i] = (ns, nd, weight, delay)
            i += 1

    return connections


def createGaussianWeights(nrPreNeurons, nrPostNeurons, mean, stdev):
    return np.random.normal(mean, stdev, (nrPreNeurons, nrPostNeurons))


def createUniformConnections(nrPreNeurons, nrPostNeurons, wMax, wMin, delay=1):
    nrConnections = nrPreNeurons * nrPostNeurons
    connections = [0] * nrConnections

    i = 0
    for ns in range(nrPreNeurons):
        for nd in range(nrPostNeurons):
            weight = abs(r.uniform(wMin, wMax))
            connections[i] = (ns, nd, weight, delay)
            i += 1

    return connections


def createUniformWeights(nrPreNeurons, nrPostNeurons, wMax, wMin):
    size = (nrPreNeurons, nrPostNeurons)
    return np.random.uniform(wMin, wMax, size)


def createConnectionsFromWeights(weights, delay=1):
    nrConnections = len(weights) * len(weights[0])  # since all neurons have the same number of connections,
    # it doesnt really matter which element we take
    connections = [0] * nrConnections
    i = 0
    for ns in range(len(weights)):  # for every source neuron
        for nd in range(len(weights[0])):  # for every dest neuron
            weight = weights[ns][nd]
            connections[i] = (ns, nd, weight, delay)
            i += 1
    return connections


def randomiseDelays(distr, connections, mean=0.5, std=0.15, wMax=1, wMin=0):
    newconn = []
    if distr == 'gaussian':
        newconn = [(conn[0], conn[1], conn[2], abs(r.gauss(mean, std))) for conn in connections]
    elif distr == 'uniform':
        for conn in connections:
            newconn = [(conn[0], conn[1], conn[2], abs(r.gauss(mean, std))) for conn in connections]
    else:
        raise ValueError("Unknown distribution for randomising delays: " + str(distr))

    return newconn


def orderRandomly(aedata1, aedata2, iterations):
    # TODO extend to handle a list of inputs
    # randomize order
    ordering = [0] * 2 * iterations
    counter1 = 0
    counter2 = 0
    for i in range(len(ordering)):
        rd = r.random()
        if (rd < 0.5 or counter2 >= iterations) and counter1 < iterations:
            ordering[i] = aedata1
            counter1 += 1

        elif counter2 < iterations:
            ordering[i] = aedata2
            counter2 += 1

    return ordering

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                           EVALUATION
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




    # def evaluate(sources, weights, delays=1):
    #     #todo input layer chold have fixed size
    #     #TODO test, write functions to save/load weights, figure out how to release spinnaker board
    #     with NeuralNet() as net:
    #         results = net.setUpEvaluation(weights, sources[0], sources[2], delays)
    #         runTime = results['runTime']
    #         net.run(runTime)
    #         net.plotSpikes(results['outputLayer'])
