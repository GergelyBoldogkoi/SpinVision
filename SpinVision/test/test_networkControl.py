import unittest
import pyNN.spiNNaker as p
import SpinVision.neuralNet as n
import SpinVision.AEDAT_Handler as f
import SpinVision.networkControl as control
import os

basePath = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/"


class networkControlTests(unittest.TestCase):
    neuronType = p.IF_curr_exp
    neuronParameters = {
        'cm': 1.0,  # The capacitance of the LIF neuron in nano-Farads
        'tau_m': 20.0,  # The time-constant of the RC circuit, in milliseconds
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

    def test_canTrainFromFile(self):
        path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/"
        files = []
        files.append(path + "10xtestSampleLeft")
        files.append(path + "10xtestSampleRight")

        weightPath = "/home/kavits/Project/SpinVision/SpinVision/resources/NetworkWeights/test/"


        out2 = control.trainFromFile(1024, 40, 1, 100, files[0], files[1])

        print out2['trainedWeights']
        print len(out2['trainedWeights'] )

        #control.saveWeights(out2['trainedWeights'], weightPath + "fullNetworkWeights")

    def test_canTrainWithWeights(self):
        path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/"
        files = []
        files.append(path + "10xtestSampleLeft")
        files.append(path + "10xtestSampleRight")

        weightPath = "/home/kavits/Project/SpinVision/SpinVision/resources/NetworkWeights/test/"
        sourceFile =  weightPath + "fullNetworkWeights"

        weights = control.loadWeights(sourceFile)

        out = control.trainWithWeights(weights, 6, 10, files[0], files[1])
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

    def test_canTrain(self):
        path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/"
        files = []
        files.append(path + "10xtestSampleLeft")
        files.append(path + "10xtestSampleRight")

        weights = control.train(1024, 40, 200, files[0], files[1], plot=True)









