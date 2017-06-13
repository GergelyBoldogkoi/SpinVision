import unittest
import pyNN.spiNNaker as p
import SpinVision.neuralNet as n
import SpinVision.AEDAT_Handler as f
import SpinVision.networkControl as control
import os
import numpy as np
import paer

basePath = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/"


class networkControlTests(unittest.TestCase):
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

    def test_canTrainFromFile(self):
        path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/"
        files = []
        files.append(path + "10xtestSampleLeft")
        files.append(path + "10xtestSampleRight")

        weightPath = "/home/kavits/Project/SpinVision/SpinVision/resources/NetworkWeights/test/"

        out2 = control.train_TrajectoriesFromFile(1024, 40, 1, 100, [files[0], files[1]])

        print out2['trainedWeights']
        print len(out2['trainedWeights'])

        # control.saveWeights(out2['trainedWeights'], weightPath + "fullNetworkWeights")

    def test_canTrainWithWeights(self):
        path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/"
        files = []
        files.append(path + "10xtestSampleLeft")
        files.append(path + "10xtestSampleRight")

        weightPath = "/home/kavits/Project/SpinVision/SpinVision/resources/NetworkWeights/test/"
        sourceFile = weightPath + "fullNetworkWeights_additiveGaussian"

        weights = control.loadWeights(sourceFile)

        out = control.train_TrajectoriesWithWeights(weights, 2, [files[0], files[1]], plot=True)
        trainedWeights = out['trainedWeights']
        unrolledWeights = [w for neuron in weights for w in neuron]
        flatTrainedWeights = [w for neuron in trainedWeights for w in neuron]

        self.assertEquals(len(unrolledWeights), len(flatTrainedWeights))

    def test_canWriteAndLoadWeights(self):
        path = "/home/kavits/Project/SpinVision/SpinVision/resources/NetworkWeights/test/"
        files = []
        files.append(path + "testWeight")

        weights = [[0.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        control.saveWeights(weights, files[0])

        loadedWeight = control.loadWeights(files[0])
        flattenedWeight = [w for neuron in weights for w in neuron]
        flatLoadedWeight = [w for neuron in loadedWeight for w in neuron]

        self.assertEquals(flatLoadedWeight, flattenedWeight)

        os.remove(files[0])

    def test_canGetNetworkResponses(self):
        path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/"
        files = []
        files.append(path + "10xtestSampleLeft")
        files.append(path + "10xtestSampleRight")

        path = "/home/kavits/Project/SpinVision/SpinVision/resources/NetworkWeights/test/"

        unTrainedWeights = control.loadWeights(path + "testUntrainedGaussian_1024x40")
        trainedWeights = control.loadWeights(path + "fullNetworkWeights_additiveGaussian")

        control.get2LayerNetworkResponses(unTrainedWeights, trainedWeights, [files[0], files[1]])

    def test_canEvaluate(self):
        path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/"
        files = []
        files.append(path + "10xtestSampleLeft")
        files.append(path + "10xtestSampleRight")

        path = "/home/kavits/Project/SpinVision/SpinVision/resources/NetworkWeights/test/"
        files.append(path + "testUntrained_1024x40")
        files.append(path + "fullNetworkWeights")

        control.evaluate(files[0], files[1], files[2], files[3])

    def test_canTrainWithWeightSource(self):
        path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/"
        files = []
        files.append(path + "10xtestSampleLeft")
        files.append(path + "10xtestSampleRight")
        path = "/home/kavits/Project/SpinVision/SpinVision/resources/NetworkWeights/test/"
        files.append(path + "testUntrainedGaussian_1024x40")
        # this is not gonna plot, just there to see if an error is raised
        weights = control.train_Trajectories(1024, 40, 2, [files[0], files[1]], plot=True, weightSource=files[2])

    def test_canPlotSpikes(self):
        untrainedSpikes = np.array([[0, 1], [1, 1], [2, 1]])
        trainedSpikes = np.array([[0, 4], [1, 4], [2, 4]])

        control.plotSpikes(untrainedSpikes, trainedSpikes)

        control.plotSpikes([],[])

    def test_can2DPlot(self):
        control.plot2DWeightsOrdered([[0.1, 0.1, 0.8, 0.7], [0.1, 0.1, 0.765, 0]],
                                     [[0.512, 0.566, 0.466, 0.42], [0.39, 0.543, 0.68, 0.01]])

    def test_canGetChangeInWeights(self):
        weights = [[1, 2, 3], [3, 4, 5]]
        trained = [[2, 2, 2], [4, 4, 4]]

        change = control.getAvgChangeInWeights(weights, trained)
        self.assertEquals(float(4.0 / 6.0), change)

    def test_canCountNeurons(self):
        data = paer.aedata()

        data.x = np.array([1, 2, 3, 4, 1, 1])
        data.y = np.array([1, 2, 3, 4, 2, 1])

        nrNeurons = control.countNeurons(data=data)

        self.assertEquals(5, nrNeurons)

    def test_findMostSpikingNeuron(self):
            spikes = [[0,2],[], [0,1,2,3]]

            nid = control.findMostSpikingNeuron(spikes)

            self.assertEquals(2, nid)

    def test_canPairInputsWithNeurons(self):
        path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/TrainingSamples/"
        files = []
        files.append(path + "Pos1-1_Sample1_denoised_32x32")  # These WORK!!
        files.append(path + "Pos5-5_Sample2_denoised_32x32")
        path2 = "/home/kavits/Project/SpinVision/SpinVision/resources/NetworkWeights/"
        files.append(path2 + "1024x10_20iter_2traj")

        weights = control.loadWeights(files[2])
        sources = [files[0], files[1]]

        pairings = control.pairInputsWithNeurons(sources, weights)

        print pairings

        self.assertEquals(0, pairings[path + "Pos1-1_Sample1_denoised_32x32"])
        self.assertEquals(1, pairings[path + "Pos5-5_Sample2_denoised_32x32"])

    def test_canTrainEndPositions(self):
        #TODO improve tests
        nrEndPositions = 2

        path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/TrainingSamples/"
        files = []
        files.append(path + "Pos1-1_Sample1_denoised_32x32")  # These WORK!!
        files.append(path + "Pos5-5_Sample2_denoised_32x32")
        path2 = "/home/kavits/Project/SpinVision/SpinVision/resources/NetworkWeights/"
        files.append(path2 + "1024x10_20iter_2traj")

        weights = control.loadWeights(files[2])

        net = control.train_endPositions(2, [files[0], files[1]], weights)

        self.assertEquals(2, len(net.layers))
        self.assertEquals(1, len(net.connections))

    def test_canPairNeuronsToPositions(self):
        #neuron trajectory pairings
        ntP = {}
        ntP['1'] = 1
        ntP['2'] = 1
        ntP['3'] = 2
        ntP['4'] = 3
        ntP['5'] = 3
        # trajectory position pairings
        tpP = {}
        tpP['1'] = 1
        tpP['2'] = 1
        tpP['3'] = 1
        tpP['4'] = 2
        tpP['5'] = 3

        npP = control.pairNeuronsToPositions(ntP, tpP)

        #there should be an error thrown: ERROR a neuron represents a wrong endposition, it has been trained to represent position 3 so it can't represent pos: 2 as well!

        self.assertEquals([1,2], npP[1])
        self.assertEquals([3], npP[2])











