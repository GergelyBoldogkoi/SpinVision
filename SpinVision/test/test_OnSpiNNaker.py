import unittest
import pyNN.spiNNaker as p
import SpinVision.neuralNet as n

class integrationTests(unittest.TestCase):
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