import pyNN.spiNNaker as p
import matplotlib.pyplot as plt
import pylab
from collections import namedtuple
Layer = namedtuple("Layer", "pop nType nParams")
Connection = namedtuple("Connection", "proj pre post type target")

# inputFormat: layerDescriptors = [[pop L1, Neuron Type L1, Neuron Params L1],[Pop L2,...]...]
class NeuralNet(object):
    def __init__(self, timestep=1):
        p.setup(timestep=1)
        self.layers = []
        self.layerTypes = []
        self.connections = []
        self.inputSpikes = []
        self.runTime = None


    # Adds a default population of neurons to the network
    def addLayerBasicLayer(self, size):
        label = str(len(self.layers))
        pop = p.Population(size, p.IF_curr_exp, {})
        layer = Layer(pop, p.IF_curr_exp, {})

        self.layers.append(layer)
        return len(self.layers) - 1

    # Adds a population with pre-determined spike-times to the Network
    def addInputLayer(self, size, sources, target='excitatory'):
        pop = p.Population(size, p.SpikeSourceArray, {'spike_times' : sources})
        layer = Layer(pop, p.SpikeSourceArray, {'spike_times' : sources})
        self.layers.append(layer)

        return len(self.layers) - 1

    #Connects two layers of the network
    def connect(self, preSynaptic, postSynaptic, type=p.AllToAllConnector(weights=5, delays=1), target='excitatory'):
        proj = p.Projection(self.layers[preSynaptic].pop, self.layers[postSynaptic].pop,type, target=target)
        connection = Connection(proj, preSynaptic, postSynaptic, type, target)

        self.connections.append(connection)
        return len(self.connections) - 1

    #runs simulation
    def run(self,runTime, record=True, record_v=True):
        self.runTime = runTime

        for population in self.layers:
            if(record):
                population.record()
            if(record_v):
                population.record_v()

        p.run(self.runTime)

    #plots spikes of a layer
    def plotSpikes(self, Layer, marker='o'):
        spikes = self.layers[Layer].getSpikes(compatible_output=True)

        plt.plot(spikes[:,1], spikes[:,0], ls='', marker=marker,markersize=4, ms=1)
        plt.xlim((-1, self.runTime))
        plt.ylim((-0.1,5))
        plt.xlabel('time (t)')
        plt.ylabel('neuron index')
        plt.show()




