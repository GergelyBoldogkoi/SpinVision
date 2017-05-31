import unittest
import pyNN.spiNNaker as p
import SpinVision.neuralNet as n
import SpinVision.AEDAT_Handler as f
import SpinVision.networkControl as control

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
        # net1 = n.NeuralNet(1)
        # out1 = net1.train(40,1,10, traininDirs, filterInputFiles="concat15")
        # net1.plotSpikes(out1)



        out2 = control.trainFromFile(956, 40, 5, 100, files[0], files[1], True)



        print out2['trainedWeights']
        print len(out2['trainedWeights'])