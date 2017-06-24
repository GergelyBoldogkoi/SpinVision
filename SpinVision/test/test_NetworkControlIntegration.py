import unittest
import pyNN.spiNNaker as p
import SpinVision.neuralNet as n
import SpinVision.neuralNet as n
import SpinVision.AEDAT_Handler as f
import SpinVision.networkControl as control
import os
import numpy as np
import paer
import SpinVision.training as tr
basePath = tr.filepath + "resources/DVS Recordings/test/"


class NetworkControlIntegrationTests(unittest.TestCase):
    neuronType = p.IF_curr_exp
    neuronParameters = {
        'cm': 12,  # The capacitance of the LIF neuron in nano-Farads
        'tau_m': 110,  # The time-constant of the RC circuit, in milliseconds
        'tau_refrac': 40,  # The refractory period, in milliseconds
        'v_reset': -70.0,  # The voltage to set the neuron at immediately after a spike
        'v_rest': -65,  # The ambient rest voltage of the neuron
        'v_thresh': -50,  # The threshold voltage at which the neuron will spike
        'tau_syn_E': 5.0,  # The excitatory input current decay time-constant
        'tau_syn_I': 10,  # The inhibitory input current decay time-constant
        'i_offset': 0.0  # A base input current to add each timestep
    }
    STDPParams = {
        'mean': 0.5,
        'std': 0.15,
        'delay': 1,
        'weightRule': 'multiplicative',
        'tauPlus': 50,
        'tauMinus': 60,
        'wMax': 1,
        'wMin': 0,
        'aPlus': 0.05,
        'aMinus': 0.05
    }

    def test_canPlotSpikes(self):
        with n.NeuralNet() as Network:
            pre = Network.addInputLayer(3, [[0, 2], [1, 3], [0]])
            post = Network.addBasicLayer(3)
            Network.connect(pre, post, p.OneToOneConnector(weights=5, delays=1))
            Network.run(10)
            Network.sampleTimes = [{'start': 1, 'end': 2}]
            Network.plotSpikes(post, block=False)

    def test_canTrainFromFile(self):
        path = tr.filepath + "resources/DVS Recordings/test/"
        files = []
        files.append(path + "10xtestSampleLeft")
        files.append(path + "10xtestSampleRight")

        weightPath = tr.filepath + "resources/NetworkWeights/test/"

        out2 = control.train_TrajectoriesFromFile(1024, 40, 1, 100, [files[0], files[1]])


    def test_canTrainWithWeights(self):
        path = tr.filepath + "resources/DVS Recordings/test/"
        files = []
        files.append(path + "10xtestSampleLeft")
        files.append(path + "10xtestSampleRight")

        weightPath = tr.filepath + "resources/NetworkWeights/test/"
        sourceFile = weightPath + "fullNetworkWeights_additiveGaussian"

        weights = control.loadWeights(sourceFile)

        out = control.train_TrajectoriesWithWeights(weights, 2, [files[0], files[1]], plot=False)
        trainedWeights = out['trainedWeights']
        unrolledWeights = [w for neuron in weights for w in neuron]
        flatTrainedWeights = [w for neuron in trainedWeights for w in neuron]

        self.assertEquals(len(unrolledWeights), len(flatTrainedWeights))

    def test_canGetNetworkResponses(self):
        path = tr.filepath + "resources/DVS Recordings/test/"
        files = []
        files.append(path + "10xtestSampleLeft")
        files.append(path + "10xtestSampleRight")

        path = tr.filepath + "resources/NetworkWeights/test/"

        unTrainedWeights = control.loadWeights(path + "testUntrainedGaussian_1024x40")
        trainedWeights = control.loadWeights(path + "fullNetworkWeights_additiveGaussian")

        control.get2LayerNetworkResponses(unTrainedWeights, trainedWeights, None, [files[0], files[1]])



    def test_canTrainWithWeightSource(self):
        path = tr.filepath + "resources/DVS Recordings/test/"
        files = []
        files.append(path + "10xtestSampleLeft")
        files.append(path + "10xtestSampleRight")
        path = tr.filepath + "resources/NetworkWeights/test/"
        files.append(path + "testUntrainedGaussian_1024x40")
        # this is not gonna plot, just there to see if an error is raised
        weights = control.train_Trajectories(1024, 40, 2, [files[0], files[1]], plot=True, weightSource=files[2])


    def test_canPairInputsWithNeurons(self):
        path = tr.filepath + "resources/DVS Recordings/TrainingSamples/"
        files = []
        files.append(path + "Pos1-1_Sample1_denoised_32x32")  # These WORK!!
        files.append(path + "Pos5-5_Sample2_denoised_32x32")
        path2 = tr.filepath + "resources/NetworkWeights/"
        files.append(path2 + "1024x10_20iter_2traj")

        weights = control.loadWeights(files[2])
        sources = [files[0], files[1]]

        pairings = control.pairInputsWithNeurons(sources, weights)

        # print pairings

        self.assertEquals(0, pairings[path + "Pos1-1_Sample1_denoised_32x32"])
        self.assertEquals(1, pairings[path + "Pos5-5_Sample2_denoised_32x32"])

    def test_canTrainEndPositions(self):
        nrEndPositions = 2

        path = tr.filepath + "resources/DVS Recordings/TrainingSamples/"
        files = []
        files.append(path + "Pos1-1_Sample1_denoised_32x32")  # These WORK!!
        files.append(path + "Pos5-5_Sample2_denoised_32x32")
        path2 = tr.filepath + "resources/NetworkWeights/"
        files.append(path2 + "1024x10_20iter_2traj")

        weights = control.loadWeights(files[2])
        tpPairings = {}
        tpPairings[files[0]] = 1
        tpPairings[files[1]] = 5

        di = control.train_endPositions(2, [files[0], files[1]], weights, tpPairings)
        net = di['net']
        pair = di['pairings']

        self.assertEquals(2, len(net.layers))
        self.assertEquals(2, len(net.connections))