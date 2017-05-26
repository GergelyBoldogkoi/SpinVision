import unittest
import pyNN.spiNNaker as p
import SpinVision.neuralNet as n
import matplotlib.pyplot as plt
import SpinVision.AEDAT_Handler as f

class neuralNetTests(unittest.TestCase):
    neuronType = p.IF_curr_exp
    neuronParameters = {
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
    def test_canCreateNeuralNet(self):
        Network =n.NeuralNet()

    def test_canAddBasicLayer(self):
        Network = n.NeuralNet()
        layerId = Network.addLayerBasicLayer(1024)

        assert 0 == layerId
        assert 1 == len(Network.layers)
        assert p.IF_curr_exp == Network.layers[0].nType
        assert {} == Network.layers[0].nParams

    def test_canAddInputLayer(self):
        Network = n.NeuralNet()
        layerId = Network.addInputLayer(2, [[0, 1], [0, 3]])

        assert 0 == layerId
        assert 1 == len(Network.layers)
        assert p.SpikeSourceArray == Network.layers[0].nType
        assert 2 == len(Network.layers[0].nParams.get('spike_times'))


    def test_canConnectLayers(self):
        Network = n.NeuralNet()
        lID1 = Network.addLayerBasicLayer(2)
        lID2 = Network.addLayerBasicLayer(2)

        c1 = Network.connect(lID1,lID2)

        assert 0 == c1
        assert lID1 == Network.connections[c1].pre
        assert lID2 == Network.connections[c1].post
        assert 'excitatory' == Network.connections[c1].type

        lID1 = Network.addLayerBasicLayer(2)
        lID2 = Network.addLayerBasicLayer(2)

        c2 = Network.connect(lID1, lID2, p.OneToOneConnector(weights=5, delays=1), 'inhibitory')
        assert 1 == c2
        assert lID1 == Network.connections[c2].pre
        assert lID2 == Network.connections[c2].post
        assert 'inhibitory' == Network.connections[c2].type

    def test_canConnectSTDP(self):
        Network = n.NeuralNet()
        lID1 = Network.addLayerBasicLayer(2)
        lID2 = Network.addLayerBasicLayer(2)

        c = Network.connectWithSTDP(lID1, lID2)

        assert 0 == c
        assert lID1 == Network.connections[c].pre
        assert lID2 == Network.connections[c].post
        assert 'STDP' == Network.connections[c].type

    def test_canPlotSpikes(self):
        Network = n.NeuralNet()
        pre = Network.addInputLayer(3,[[0, 2], [1, 3], [0]])
        post = Network.addLayerBasicLayer(3)
        Network.connect(pre, post, p.OneToOneConnector(weights=5, delays=1))
        Network.run(10)
        Network.plotSpikes(post, block=False)

    def test_canReadSpikes(self):

        Network = n.NeuralNet()

        fileName = "/home/kavits/Project/SpinVision/SpinVision/resources/" \
                   "DVS Recordings/test/testTruncated"
        spikeTimes = Network.readSpikes(fileName)

        flattenedList = [timeStamp for neuron in spikeTimes for timeStamp in neuron]
        nrSpikes = len(flattenedList)

        nrSpikesInFile = len(f.extractData(f.readData(fileName))['ts'])

        assert nrSpikesInFile == nrSpikes
        assert 937 == len(spikeTimes)
