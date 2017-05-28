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
    def __init__(self, timestep=1):
        p.setup(timestep=1)
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

    #adds layer to network
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

    # trains 2 layers
    def setup2Layer(self, outpuLayerSize, trainingDir, timeBetweenSamples, STDP_Params, neuronParams, neuronType=p.IF_curr_exp, filterInputFiles=None, save=False,destPath=None):

        if (not len(self.layers) == 0) and (not len(self.connections) == 0):
            raise TypeError(
                "Network has already been initialized, please ensure to call this function on an uninitialized network")

        trainingSpikes = getTrainingData(trainingDir, filterInputFiles, timeBetweenSamples, 0, save, destPath)

        inputLayerNr = self.addInputLayer(len(trainingSpikes),trainingSpikes)

        outputLayerNr = self.addLayer(outpuLayerSize, neuronType, neuronParams)

        stdpNr = self.connectWithSTDP(inputLayerNr, outputLayerNr,
                             STDP_Params['mean'], STDP_Params['std'], STDP_Params['delay'],
                             STDP_Params['weightRule'], STDP_Params['tauPlus'],
                             STDP_Params['tauMinus'], STDP_Params['wMax'],
                             STDP_Params['wMin'], STDP_Params['aPlus'], STDP_Params['aMinus'])

        inhibitoryNr = self.connect(outputLayerNr, outputLayerNr,
                                    connectivity=p.AllToAllConnector(weights=5, delays=STDP_Params['delay']),
                                    type='inhibitory')

        return {'inputLayer': inputLayerNr, 'outputLayer': outputLayerNr,
                'inhibitoryConnection': inhibitoryNr, 'STDPConnection': stdpNr}

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                           OUTPUT
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # plots spikes of a layer
    def plotSpikes(self, layerId, marker='o', block=True):
        spikes = self.layers[layerId].pop.getSpikes(compatible_output=True)

        plt.plot(spikes[:, 1], spikes[:, 0], ls='', marker=marker, markersize=4, ms=1)
        plt.xlim((-1, self.runTime))
        plt.ylim((-0.1, 5))
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


def readSpikes(aedata, startFrom_ms=None, convertTo_ms=True):
    data = f.extractData(aedata)

    organisedData = {}  # datastructure containing a structure of spiketimes for each neuron
    # a new neuron is created for each x,y and ONOFF value
    shift = 0  # is in us
    startsAt = data['ts'][0]
    spikeTime = 0

    # find out how much to shift by such that the first spike is at starFrom_ms
    if startFrom_ms is not None:
        shift = startsAt - startFrom_ms * 1000  # assumes data is ordered according to spiketimes
        startsAt = startsAt - shift
        if convertTo_ms:
            startsAt /= 1000
    for i in range(len(data['ts'])):
        neuronId = (data['X'][i], data['Y'][i], data['t'][i])  # x,y,ONOFF

        spikeTime = data['ts'][i] - shift

        if convertTo_ms:
            spikeTime /= 1000

        if neuronId not in organisedData:

            organisedData[neuronId] = [spikeTime]

        else:
            organisedData[neuronId].append(spikeTime)

    endsAt = spikeTime
    return organisedData


# This function reads in spikes from all files in given directories
def getTrainingData(trainingDirectories, filter=None, timeBetweenSamples=0, startFrom_ms=0, save=False,
                    destFile=None):
    # type: ([str], str, int, int, bool, str) -> [[float]]

    aedata = f.concatenate(trainingDirectories, timeBetweenSamples * 1000, filter, destFile, save)

    data = readSpikes(aedata, startFrom_ms)
    spikeTimes = []
    for neuronSpikes in data.values():
        neuronSpikes.sort()
        spikeTimes.append(neuronSpikes)

    return spikeTimes


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
