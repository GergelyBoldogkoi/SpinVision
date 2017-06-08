import AEDAT_Handler as f
import neuralNet as n
import pyNN.spiNNaker as p
import random as r
import networkControl as control


def trainWithWeightSource():

    path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/TrainingSamples/"
    files = []
    files.append(path + "Pos1-1_Sample1_denoised_32x32")
    files.append(path + "Pos5-5_Sample2_denoised_32x32")
    path = "/home/kavits/Project/SpinVision/SpinVision/resources/NetworkWeights/test/"
    #files.append(path + "testUntrainedGaussian_1024x40")
    # this is not gonna plot, just there to see if an error is raised

    print "neuronType: " + str(n.__neuronType__)
    print "STDPParameters: " + str(n.__STDPParameters__)
    print "Neuron Params: " + str(n.__neuronParameters__)

    weights = control.trainTrajectories(100, 2, 20, files[0], files[1], plot=True)
    # weights = control.trainTrajectories(60, 3, 100, files[0], files[1], plot=True)#, weightSource=files[2])
    print weights

trainWithWeightSource()