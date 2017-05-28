import pyNN.spiNNaker as p
import matplotlib.pyplot as plt
import pylab
from collections import namedtuple
import AEDAT_Handler as f
from os import listdir

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
    def addLayerBasicLayer(self, size):
        label = str(len(self.layers))
        pop = p.Population(size, p.IF_curr_exp, {})
        layer = Layer(pop, p.IF_curr_exp, {})

        self.layers.append(layer)
        return len(self.layers) - 1

    # Adds a population with pre-determined spike-times to the Network
    def addInputLayer(self, size, sources, type='excitatory'):
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

    def connectWithSTDP(self, pre, post, connectivity=p.AllToAllConnector(weights=0.5, delays=1),
                        weightMod='additive',
                        tauPlus=20, tauMinus=20,
                        wMax=1, wMin=0,
                        aPlus=0.5, aMinus=0.5):
        # Maybe extend to save parameters of STDP Connection
        timingRule = p.SpikePairRule(tau_plus=tauPlus, tau_minus=tauMinus)

        # -------- Setting Weight Rule ------------
        weightRule = None
        if weightMod == 'additive':
            weightRule = p.AdditiveWeightDependence(w_max=wMax, w_min=wMin, A_plus=0.5, A_minus=aMinus)
        # TODO add multiplicative weightrule
        else:
            raise TypeError(str(weightMod) + " is not a known weight modification rule for STDP \n"
                                             "try \'additive\' or \'multiplicative\'")
        # ------------------------------------------

        STDP_Model = p.STDPMechanism(timing_dependence=timingRule, weight_dependence=weightRule)

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

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                           TRAINING
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # this function returns a list of spike timings read from a file
    # ASSUMES .AEDAT FILES TO BE ORDERED ACCORDING TO TIMESTAMPS!!!!!

    def readSpikes(self, aedata, startFrom_ms=None, convertTo_ms=True):

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
    def getTrainingData(self, trainingDirectories, filter=None, timeBetweenSamples=0, startFrom_ms=0, save=False, filename=None):

        aedata = f.append(trainingDirectories, timeBetweenSamples*1000, filter, filename, save)

        data = self.readSpikes(aedata, startFrom_ms)
        # startAt = startFrom_ms
        # for dir in trainingDirectories:
        #     for file in listdir(dir):  # append spikes from new file to existing
        #         filePath = dir + "/" + file[0:len(file) - len(".aedat")]  # get rid of ".aedat"
        #         spikeCont = self.readSpikes(filePath, startFrom_ms=startAt)
        #
        #         startAt += spikeCont['timeSpan'] + timeBetweenSamples  # prepare to append data from next file
        #
        #         data = spikeCont['data']
        #
        #         collectiveData = {neuron: collectiveData.get(neuron, []) + data.get(neuron, []) for neuron in
        #                           set(collectiveData) | set(data)}

        spikeTimes = []
        for neuronSpikes in data.values():
            neuronSpikes.sort()
            spikeTimes.append(neuronSpikes)

        return spikeTimes

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
