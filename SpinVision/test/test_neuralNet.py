import unittest
import pyNN.spiNNaker as p
import SpinVision.neuralNet as n
import matplotlib.pyplot as plt
import SpinVision.AEDAT_Handler as f
from os import listdir
import paer
import SpinVision.training as tr

basePath = tr.filepath + "resources/DVS Recordings/test/"


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
    STDPParams = {
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

    def test_canCreateNeuralNet(self):
        Network = n.NeuralNet()

    def test_canAddBasicLayer(self):
        Network = n.NeuralNet()
        layerId = Network.addBasicLayer(1024)

        assert 0 == layerId
        assert 1 == len(Network.layers)
        assert p.IF_curr_exp == Network.layers[0].nType
        assert {} == Network.layers[0].nParams

    def test_canAddLayer(self):
        neuronType = p.IF_curr_dual_exp
        Network = n.NeuralNet()
        layerId = Network.addLayer(1024, neuronType, self.neuronParameters)

        self.assertEquals(0, layerId)
        self.assertTrue(Network.layers[0].nType is neuronType)
        self.assertTrue(Network.layers[0].nParams is self.neuronParameters)

    def test_canAddInputLayer(self):
        Network = n.NeuralNet()
        layerId = Network.addInputLayer(2, [[0, 1], [0, 3]])

        assert 0 == layerId
        assert 1 == len(Network.layers)
        assert p.SpikeSourceArray == Network.layers[0].nType
        self.assertEquals(2 , len(Network.layers[0].nParams.get('spike_times')))

    def test_canConnectLayers(self):
        Network = n.NeuralNet()
        lID1 = Network.addBasicLayer(2)
        lID2 = Network.addBasicLayer(2)

        c1 = Network.connect(lID1, lID2)

        assert 0 == c1
        assert lID1 == Network.connections[c1].pre
        assert lID2 == Network.connections[c1].post
        assert 'excitatory' == Network.connections[c1].type

        lID1 = Network.addBasicLayer(2)
        lID2 = Network.addBasicLayer(2)

        c2 = Network.connect(lID1, lID2, p.OneToOneConnector(weights=5, delays=1), 'inhibitory')
        assert 1 == c2
        assert lID1 == Network.connections[c2].pre
        assert lID2 == Network.connections[c2].post
        assert 'inhibitory' == Network.connections[c2].type

    def test_canConnectSTDP(self):
        Network = n.NeuralNet()
        lID1 = Network.addBasicLayer(2)
        lID2 = Network.addBasicLayer(2)

        c = Network.connectWithSTDP(lID1, lID2)

        assert 0 == c
        assert lID1 == Network.connections[c].pre
        assert lID2 == Network.connections[c].post
        assert 'STDP' == Network.connections[c].type

        lID1 = Network.addBasicLayer(2)
        lID2 = Network.addBasicLayer(2)

        c = Network.connectWithSTDP(lID1, lID2, weightMod='multiplicative')

        assert 1 == c
        assert lID1 == Network.connections[c].pre
        assert lID2 == Network.connections[c].post
        assert 'STDP' == Network.connections[c].type



    def test_canReadSpikes(self):
        Network = n.NeuralNet()

        filename = tr.filepath + "resources/" \
                   "DVS Recordings/test/testTruncated"
        aedata = f.readData(filename)
        ahham = n.readSpikes([aedata])
        spikeTimes = ahham['data']
        lastSpike = ahham['lastSpikeAt']

        flattenedList = [timeStamp for neuron in spikeTimes.values() for timeStamp in neuron]
        nrSpikes = len(flattenedList)

        nrSpikesInFile = len(f.extractData(f.readData(filename))['ts'])

        self.assertEquals(nrSpikesInFile, nrSpikes)
        # print len(spikeTimes)
        self.assertEquals(664, len(spikeTimes))

        lastSpikeInFile = (aedata.ts[len(aedata.ts) - 1] - aedata.ts[0]) / 1000
        self.assertEquals(lastSpikeInFile, lastSpike)

        # check start time works
        startTime = 0
        hab = n.readSpikes([aedata])
        spikeTimes = hab['data']
        lastSpike = hab['lastSpikeAt']
        flattenedList = [timeStamp for neuron in spikeTimes.values() for timeStamp in neuron]

        flattenedList.sort()
        self.assertEquals(flattenedList[0], startTime
                          , "check whether list actually starts from 0")

    def test_canGetTrainingDataFromDirectories(self):
        Network = n.NeuralNet()
        path = tr.filepath + "resources/DVS Recordings/test/appendTest"

        tbs = 1000
        data = n.getTrainingDataFromDirectories([path], filter="test", timeBetweenSamples=tbs, startFrom_ms=0)[
            'spikeTimes']
        flattenedList = [timeStamp for neuron in data for timeStamp in neuron]
        nrSpikes = len(flattenedList)

        nrSpikesControl = 0
        for file in listdir(path):
            # print file
            if "notTobeIncludedInAppend.aedat" == file:  # make sure filtering works
                print "ignoring incorrect file"
                continue

            nrSpikesControl += len(f.extractData(f.readData(path + "/" + file[0:len(file) - len(".aedat")]))['ts'])

        self.assertEquals(nrSpikesControl, nrSpikes)

    def test_canCreateGaussianConnections(self):

        mean = 0.5
        nrSource = 100
        nrDest = 100
        connections = n.createGaussianConnections(nrSource, nrDest, mean, 0.15)

    def test_canCreateUniformConnections(self):
        wMax = 1
        wMin = 0
        nrSource = 100
        nrDest = 100
        connections = n.createUniformConnections(nrSource, nrDest, wMax, wMin)

        self.assertEquals(nrSource * nrDest, len(connections))

    def test_canCreateUniformWeights(self):

        weights = n.createUniformWeights(1024, 16, 1, 0)

    # def test_canRandomoseDelays(self):
    #     wMax = 1
    #     wMin = 0
    #     nrSource = 100
    #     nrDest = 100
    #     connections = n.createUniformConnections(nrSource, nrDest, wMax, wMin)
    #     print "print yoo"
    #     newConn = n.randomiseDelays('gaussian', connections)
    #     print "print yoo"
    #
    #     self.assertEquals(len(connections), len(newConn))
    #
    #     newConn = n.randomiseDelays('uniform', connections)
    #     self.assertEquals(len(connections), len(newConn))

    def test_canCreateConnectionsFromWeights(self):
        weights = [[0, 2, 3], [5, 6, 7]]

        connections = n.createConnectionsFromWeights(weights)

        unrolledConn = [w for neuron in weights for w in neuron]
        self.assertEquals(len(unrolledConn), len(connections))

        i = 0
        for ns in range(len(weights)):
            for nd in range(len(weights[0])):
                self.assertEquals(weights[ns][nd], connections[i][2])
                self.assertEquals(ns, connections[i][0])
                self.assertEquals(nd, connections[i][1])
                self.assertEquals(1, connections[i][3])
                i += 1

    def test_canGetTrainingData(self):
        Network = n.NeuralNet()
        path = tr.filepath + "resources/DVS Recordings/test/"

        files = []
        files.append(path + "testSampleLeft")
        files.append(path + "testSampleRight")

        returned = n.getTrainingData(16384, [files[0], files[1]], 2, 100)
        self.assertEquals(16384, len(returned['spikeTimes']))
        # print returned['sampleTimes']
        # The tests here would really be the same as for readSpikes

        # spikeTimes=[]
        # layerWidth = 4
        # for x in range(layerWidth):
        #     for y in range(layerWidth):
        #         spikeTimes.append([])
        # print spikeTimes

    def test_canSetupForInitialTraining(self):
        Network = n.NeuralNet()
        path =tr.filepath + "resources/DVS Recordings/test/"

        files = []
        files.append(path + "testSampleLeft")
        files.append(path + "testSampleRight")

        hab = Network.setUp2LayersForTraining(16384, 40, [files[0], files[1]], 100,
                                              iterations=2)

    def test_canSetupWithWeights(self):
        with n.NeuralNet() as net:
            path = tr.filepath + "resources/DVS Recordings/test/"

            files = []
            files.append(path + "testSampleLeft")
            files.append(path + "testSampleRight")

            weights = [[0., 1., 2.], [3., 4., 5.]]

            data = net.setUpTrainingWithWeights(weights, [files[0], files[1]])
            net.run(10)

            unrolledWeights = [w for neuron in weights for w in neuron]
            stw = net.connections[0].proj.getWeights(format='array')
            unrolledStW = [w for neuron in stw for w in neuron]

            self.assertEquals(len(unrolledStW), len(unrolledWeights),
                              str(len(unrolledStW)) + " != " + str(len(unrolledWeights)) + " weights set incorrecty!")

    def test_canSetupEvaluation(self):
        with n.NeuralNet() as net:
            path = tr.filepath + "resources/DVS Recordings/test/"

            files = []
            files.append(path + "testSampleLeft")
            files.append(path + "testSampleRight")

            weights = [[0., 1., 2.], [3., 4., 5.]]

            data = net.setUp2LayerEvaluation(weights, files[0], files[1])

            self.assertEquals(files[0], net.annotations[0], "annotations set in correct order")
            self.assertEquals(files[1], net.annotations[1], "annotations set in correct order")



