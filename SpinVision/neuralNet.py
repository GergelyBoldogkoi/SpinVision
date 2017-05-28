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
        'tau_refrac': 2.0,  # The refractory period, in milliseconds
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
        'std': 0.15,
        'delay': 1,
        'weightRule': 'additive',
        'tauPlus': 20,
        'tauMinus': 20,
        'wMax': 1,
        'wMin': 0,
        'aPlus': 0.5,
        'aMinus': 0.5
    }

    def __init__(self, timestep=1):
        p.setup(timestep=timestep)
        self.layers = []  # This list should contain Layers, see namedTuple def
        self.connections = []  # This list should contain Connections, see namedtuple def
        self.runTime = None

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
    def addInputLayer(self, size, sources):
        pop = p.Population(size, p.SpikeSourceArray, {'spike_times': sources})
        layer = Layer(pop, p.SpikeSourceArray, {'spike_times': sources})
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
                        aPlus=0.5, aMinus=0.5):
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

        connections = createGaussianConnections(self.layers[pre][0].size, self.layers[post][0].size,
                                                initWeightMean, initWeightStd, delay)
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
    def run(self, runTime, record=True, record_v=True):
        self.runTime = runTime

        for layer in self.layers:
            if record:
                layer.pop.record()
            if record_v:
                if 'spike_times' not in layer.nParams:  # It is not possible to record voltages for inputLayers
                    layer.pop.record_v()

        p.run(runTime)

    # sets up 2 layers for training
    # when this function save, it is not going to save the actual training data, but the one that has been
    # iterated iterations times
    def setup2Layers(self, outpuLayerSize, trainingDir, timeBetweenSamples, iterations=1,
                     filterInputFiles=None, save=False, destPath=None):

        if (not len(self.layers) == 0) and (not len(self.connections) == 0):
            raise TypeError(
                "Network has already been initialized, please ensure to call this function on an uninitialized network")

        bundle = getTrainingData(trainingDir, iterations=iterations, filter=filterInputFiles,
                                 timeBetweenSamples=timeBetweenSamples,
                                 startFrom_ms=0, save=save, destFile=destPath)
        lastSpike = bundle['lastSpikeAt']
        trainingSpikes = bundle['spikeTimes']

        inputLayerNr = self.addInputLayer(len(trainingSpikes), trainingSpikes)

        outputLayerNr = self.addLayer(outpuLayerSize, self.__neuronType__, self.__neuronParameters__)

        STDP_Params = self.__STDPParameters__
        stdpNr = self.connectWithSTDP(inputLayerNr, outputLayerNr,
                                      STDP_Params['mean'], STDP_Params['std'], STDP_Params['delay'],
                                      STDP_Params['weightRule'], STDP_Params['tauPlus'],
                                      STDP_Params['tauMinus'], STDP_Params['wMax'],
                                      STDP_Params['wMin'], STDP_Params['aPlus'], STDP_Params['aMinus'])

        inhibitoryNr = self.connect(outputLayerNr, outputLayerNr,
                                    connectivity=p.AllToAllConnector(weights=5, delays=STDP_Params['delay']),
                                    type='inhibitory')

        return {'inputLayer': inputLayerNr, 'outputLayer': outputLayerNr,
                'inhibitoryConnection': inhibitoryNr, 'STDPConnection': stdpNr,
                'lastSpikeAt': lastSpike}

    # This function trains the network
    def train(self, outputLayerSize, iterations, timeBetweenSamples, trainingDir, filterInputFiles=None, save=False,
              destPath=None):

        net = self.setup2Layers(outpuLayerSize=outputLayerSize, trainingDir=trainingDir,
                                timeBetweenSamples=timeBetweenSamples, iterations=iterations,
                                filterInputFiles=filterInputFiles, save=save, destPath=destPath)

        runTime = int(net['lastSpikeAt'] + 10)
        print str(runTime) + " this is the RunTime"
        self.run(runTime, record=True)

        outputPop = self.layers[net['outputLayer']].pop

        return net['outputLayer']

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                           OUTPUT
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # plots spikes of a layer
    def plotSpikes(self, layerId, marker='o', block=True):
        spikes = self.layers[layerId].pop.getSpikes(compatible_output=True)

        plt.plot(spikes[:, 1], spikes[:, 0], ls='', marker=marker, markersize=4, ms=1)
        plt.xlim((-1, self.runTime))
        plt.ylim((-0.1, self.layers[layerId].pop.size))
        plt.xlabel('time (t)')
        plt.ylabel('neuron index')
        plt.show(block=block)

    def getWeights(self, connection):

        return self.connections[connection][0].getWeights()


# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#                           TRAINING
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# this function returns a list of spike timings read from a file
# ASSUMES .AEDAT FILES TO BE ORDERED ACCORDING TO TIMESTAMPS!!!!!


def readSpikes(aedata, iterations=1, timeBetweenIterations=0, startFrom_ms=None):
    data = f.extractData(aedata)

    # TODO could be made more efficient and much nicer

    organisedData = {}  # datastructure containing a structure of spiketimes for each neuron
    # a new neuron is created for each x,y and ONOFF value
    shift_us = 0  # is in us
    startsAt = data['ts'][0]
    spikeTime = 0
    lastSpikeAt = 0

    # find out how much to shift by such that the first spike is at starFrom_ms
    if startFrom_ms is None:
        startFrom_ms = startsAt / 1000

    shift_us = startsAt - startFrom_ms * 1000  # assumes data is ordered according to spiketimes
    for it in range(iterations):

        for i in range(len(data['ts'])):
            neuronId = (data['X'][i], data['Y'][i], data['t'][i])  # x,y,ONOFF

            spikeTime = data['ts'][i] - shift_us

            if neuronId not in organisedData:
                organisedData[neuronId] = [spikeTime / 1000]

            else:
                organisedData[neuronId].append(spikeTime / 1000)

        shift_us -= spikeTime + timeBetweenIterations * 1000

    return {'data': organisedData, 'lastSpikeAt': spikeTime / 1000}


# This function reads in spikes from all files in given directories
def getTrainingData(trainingDirectories, filter=None, iterations=1, timeBetweenSamples=0, startFrom_ms=0, save=False,
                    destFile=None):
    # type: ([str], str, int,  int, int, bool, str) -> [[float]]

    aedata = f.concatenate(trainingDirectories, timeBetweenSamples * 1000, filter, destFile, save)

    cont = readSpikes(aedata, iterations, timeBetweenSamples, startFrom_ms)
    data = cont['data']
    finishesAt = cont['lastSpikeAt']
    spikeTimes = []
    for neuronSpikes in data.values():
        neuronSpikes.sort()
        spikeTimes.append(neuronSpikes)

    return {'spikeTimes': spikeTimes, 'lastSpikeAt': finishesAt}


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
