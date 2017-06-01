import pyNN.spiNNaker as p
import matplotlib.pyplot as plt
import pylab
from collections import namedtuple
import AEDAT_Handler as f
import random as r

Layer = namedtuple("Layer", "pop nType nParams")
Connection = namedtuple("Connection", "proj pre post connectivity type")


# inputFormat: layerDescriptors = [[pop L1, Neuron Type L1, Neuron Params L1],[Pop L2,...]...]



class NeuralNet(object):
    __neuronParameters__ = {
        'cm': 1.0,  # The capacitance of the LIF neuron in nano-Farads
        'tau_m': 20.0,  # The time-constant of the RC circuit, in millisecon
        'tau_refrac': 20.0,  # The refractory period, in milliseconds
        'v_reset': -70.0,  # The voltage to set the neuron at immediately after a spike
        'v_rest': -65,  # The ambient rest voltage of the neuron
        'v_thresh': -50,  # The threshold voltage at which the neuron will spike
        'tau_syn_E': 5.0,  # The excitatory input current decay time-constant
        'tau_syn_I': 5.0,  # The inhibitory input current decay time-constant
        'i_offset': 0.0  # A base input current to add each timestep
    }
    __neuronType__ = p.IF_curr_exp

    __STDPParameters__ = {
        'mean': 0.5,
        'std': 0.75,
        'delay': 1,
        'weightRule': 'additive',
        'tauPlus': 20,
        'tauMinus': 20,
        'wMax': 1,
        'wMin': 0,
        'aPlus': 0.5,
        'aMinus': 0.5
    }

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
        pop = p.Population(size, p.SpikeSourceArray, {'spike_times': sourceSpikes})
        layer = Layer(pop, p.SpikeSourceArray, {'spike_times': sourceSpikes})
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
                        aPlus=0.5, aMinus=0.5, weights=None):
        # Maybe extend to save parameters of STDP Connection
        timingRule = p.SpikePairRule(tau_plus=tauPlus, tau_minus=tauMinus)

        # -------- Setting Weight Rule ------------
        weightRule = None
        if weightMod == 'additive':
            weightRule = p.AdditiveWeightDependence(w_max=wMax, w_min=wMin, A_plus=aPlus, A_minus=aMinus)
        # TODO add multiplicative weightrule
        else:
            raise TypeError(str(weightMod) + " is not a known weight modification rule for STDP \n"
                                             "try \'additive\' or \'multiplicative\'")
        # ------------------------------------------

        STDP_Model = p.STDPMechanism(timing_dependence=timingRule, weight_dependence=weightRule)

        #SET UP WEIGHTS OF TH ENETWORK
        if weights is None: #if there are no weighs supplied create randomised weights
            connections = createGaussianConnections(self.layers[pre][0].size, self.layers[post][0].size,
                                                initWeightMean, initWeightStd, delay)
        else:#if weights are supplied set up STDP with those
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
    def setUpForTraining(self, inputLayerSize, outpuLayerSize, source1, source2, timeBetweenSamples, iterations=1, weights=None):



        if (not len(self.layers) == 0) and (not len(self.connections) == 0):
            raise TypeError(
                "Network has already been initialized, please ensure to call this function on an uninitialized network")

        bundle = getTrainingData(inputLayerSize, source1, source2, iterations, timeBetweenSamples)
        lastSpike = bundle['lastSpikeAt']
        trainingSpikes = bundle['spikeTimes']


        inputLayerNr = self.addInputLayer(inputLayerSize, trainingSpikes)

        outputLayerNr = self.addLayer(outpuLayerSize, self.__neuronType__, self.__neuronParameters__)

        STDP_Params = self.__STDPParameters__
        stdpNr = self.connectWithSTDP(inputLayerNr, outputLayerNr,
                                      STDP_Params['mean'], STDP_Params['std'], STDP_Params['delay'],
                                      STDP_Params['weightRule'], STDP_Params['tauPlus'],
                                      STDP_Params['tauMinus'], STDP_Params['wMax'],
                                      STDP_Params['wMin'], STDP_Params['aPlus'], STDP_Params['aMinus'],
                                      weights)


        inhibitoryNr = self.connect(outputLayerNr, outputLayerNr,
                                    connectivity=p.AllToAllConnector(weights=5, delays=STDP_Params['delay']),
                                    type='inhibitory')


        return {'inputLayer': inputLayerNr, 'outputLayer': outputLayerNr,
                'inhibitoryConnection': inhibitoryNr, 'STDPConnection': stdpNr,
                'lastSpikeAt': lastSpike}


    def setUpEvaluation(self, weights, source1, source2, delay=1):
        # len(weights[0]) returns how many output neurons there are,
        # as all inputs are connected to all outputs, so it doesn't really matter
        # precisely which element of weights we take
        nrOut = len(weights[0])
        nrIn = len(weights)

        outputLayerNr = self.addLayer(nrOut, self.__neuronType__, self.__neuronParameters__)

        bundle = getTrainingData(nrIn, source1, source2, 1, 100)
        evalSpikes = bundle['spikeTimes']

        lastSpike = bundle['lastSpikeAt']

        inputLayerNr = self.addInputLayer(nrIn, evalSpikes)

        self.sampleTimes = bundle['sampleTimes']
        self.annotations.append(source1)
        self.annotations.append(source2)

        neuronConnections = [0] * nrIn*nrOut
        for i in range(nrIn):
            for j in range(nrOut):
                neuronConnections[i * nrOut + j] = (i, j, weights[i][j], delay)

        connection = self.connect(inputLayerNr, outputLayerNr, p.FromListConnector(neuronConnections), type='excitatory')

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
        sampleLength = (sample.ts[len(sample.ts) - 1] - sample.ts[0])/1000
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

    return {'data': organisedData, 'sampleTimes': sampleTimes_ms,  'lastSpikeAt': sampleEnd}


# This function reads in spikes from all files in given directories
def getTrainingDataFromDirectories(trainingDirectories, filter=None, iterations=1, timeBetweenSamples=0, startFrom_ms=0,
                                   save=False,
                                   destFile=None):
    # type: ([str], str, int,  int, int, bool, str) -> [[float]]

    aedata = f.concatenate(trainingDirectories, timeBetweenSamples * 1000, filter, destFile, save)

    cont = readSpikes([aedata],  timeBetweenIterations=timeBetweenSamples)
    data = cont['data']
    finishesAt = cont['lastSpikeAt']
    spikeTimes = []
    for neuronSpikes in data.values():
        neuronSpikes.sort()
        spikeTimes.append(neuronSpikes)

    return {'spikeTimes': spikeTimes, 'lastSpikeAt': finishesAt}


def getTrainingData(inputlayerSize, sourceFile1, sourceFile2, iterations, timebetweenSamples, randomise=True):
    aedata1 = f.readData(sourceFile1)
    aedata2 = f.readData(sourceFile2)

    if randomise:
        ordering = orderRandomly(aedata1, aedata2, iterations)
    else:
        ordering = []
        for i in range(iterations):
            ordering.append(aedata1)
            ordering.append(aedata2)


    cont = readSpikes(ordering, timebetweenSamples)
    data = cont['data']

    spikeTimes = []
    for neuronSpikes in data.values():
        neuronSpikes.sort()
        spikeTimes.append(neuronSpikes)

    fillFrom = len(spikeTimes)
    for i in range(fillFrom, inputlayerSize):
        spikeTimes.append([])

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
def createConnectionsFromWeights(weights, delay=1):
    nrConnections = len(weights) * len(weights[0])  #since all neurons have the same number of connections,
                                                    # it doesnt really matter which element we take
    connections = [0] * nrConnections
    i = 0
    for ns in range(len(weights)): #for every source neuron
        for nd in range(len(weights[0])): #for every dest neuron
            weight = weights[ns][nd]
            connections[i] = (ns, nd, weight, delay)
            i += 1
    return connections


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