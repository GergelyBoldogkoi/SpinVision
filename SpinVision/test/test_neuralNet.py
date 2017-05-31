import unittest
import pyNN.spiNNaker as p
import SpinVision.neuralNet as n
import matplotlib.pyplot as plt
import SpinVision.AEDAT_Handler as f
from os import listdir
import paer

basePath = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/"


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
        assert 2 == len(Network.layers[0].nParams.get('spike_times'))

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

    def test_canPlotSpikes(self):
        with n.NeuralNet() as Network:
            pre = Network.addInputLayer(3, [[0, 2], [1, 3], [0]])
            post = Network.addBasicLayer(3)
            Network.connect(pre, post, p.OneToOneConnector(weights=5, delays=1))
            Network.run(10)
            Network.sampleTimes = [{'start': 1, 'end': 2}]
            Network.plotSpikes(post)

    def test_canReadSpikes(self):
        Network = n.NeuralNet()

        filename = "/home/kavits/Project/SpinVision/SpinVision/resources/" \
                   "DVS Recordings/test/testTruncated"
        aedata = f.readData(filename)
        ahham = n.readSpikes([aedata])
        spikeTimes = ahham['data']
        lastSpike = ahham['lastSpikeAt']

        flattenedList = [timeStamp for neuron in spikeTimes.values() for timeStamp in neuron]
        nrSpikes = len(flattenedList)

        nrSpikesInFile = len(f.extractData(f.readData(filename))['ts'])

        self.assertEquals(nrSpikesInFile, nrSpikes)
        print len(spikeTimes)
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

        # make sure iterations and multiple samples work
        nrIterations = 2
        nrSamples = 2
        tbI = 100
        hab2 = n.readSpikes([aedata, aedata], iterations=nrIterations, timeBetweenIterations=tbI)
        spikeTimes2 = hab2['data']
        lastSpike2 = hab2['lastSpikeAt']
        flattenedList2 = [timeStamp for neuron in spikeTimes2.values() for timeStamp in neuron]
        flattenedList2.sort()

        self.assertEquals(len(flattenedList2), len(flattenedList) * nrIterations * nrSamples
                          , "Not all elements added to list with iteration")
        self.assertEquals(lastSpike * nrIterations * nrSamples + tbI * (nrSamples * nrIterations - 1), lastSpike2
                          , "LastSpikeAt not set correctly for list with iteration")

    def test_canGetTrainingDataFromDirectories(self):
        Network = n.NeuralNet()
        path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/appendTest"

        tbs = 1000
        data = n.getTrainingDataFromDirectories([path], filter="test", timeBetweenSamples=tbs, startFrom_ms=0)[
            'spikeTimes']
        flattenedList = [timeStamp for neuron in data for timeStamp in neuron]
        nrSpikes = len(flattenedList)

        nrSpikesControl = 0
        for file in listdir(path):
            print file
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

        self.assertEquals(nrSource * nrDest, len(connections))

    def test_canGetTrainingData(self):
        Network = n.NeuralNet()
        path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/"

        files = []
        files.append(path + "testSampleLeft")
        files.append(path + "testSampleRight")

        returned = n.getTrainingData(1024,files[0], files[1], 2, 100)
        self.assertEquals(1024, len(returned['spikeTimes']))
        print returned['sampleTimes']
        # The tests here would really be the same as for readSpikes

    def test_canSetupForInitialTraining(self):
        Network = n.NeuralNet()
        path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/"

        files = []
        files.append(path + "testSampleLeft")
        files.append(path + "testSampleRight")

        hab = Network.setUpInitialTraining(1024, 40, files[0], files[1], 100,
                                           iterations=2)
    def test_canSetupWithWeights(self):
        net = n.NeuralNet()
        path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/"

        files = []
        files.append(path + "testSampleLeft")
        files.append(path + "testSampleRight")

        weights = [[0., 1., 2.], [3., 4., 5.]]

        data = net.setUpTrainingWithWeights(weights, files[0], files[1])
        net.run(10)

        unrolledWeights = [w for neuron in weights for w in neuron]
        stw = net.connections[0].proj.getWeights(format='array')
        unrolledStW = [w for neuron in stw for w in neuron]

        self.assertEquals(len(unrolledStW) ,len(unrolledWeights),
                          str(len(unrolledStW)) + " != " + str(len(unrolledWeights)) + " weights set incorrecty!")

    def test_canSetupEvaluation(self):
        net = n.NeuralNet()
        path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/"

        files = []
        files.append(path + "testSampleLeft")
        files.append(path + "testSampleRight")

        weights = [[0., 1., 2.], [3., 4., 5.]]

        data = net.setUpEvaluation(weights, files[0], files[1])

        self.assertEquals(files[0], net.annotations[0], "annotations set in correct order")
        self.assertEquals(files[1], net.annotations[1], "annotations set in correct order")

        # net.run(1)
        #
        # conWeights = net.connections[0].proj.getWeights(format='array')
        #
        #
        #
        # self.assertEquals(conWeights, weights) #Todo weights work correctly, but different type numbers are returned, find a way to compare them


