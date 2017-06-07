import AEDAT_Handler as f
import neuralNet as n
import pyNN.spiNNaker as p
import random as r
import networkControl as control


def trainWithWeightSource():
    path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/TrainingSamples/16x16/"
    files = []
    files.append(path + "Pos3To1_lowAngle_16x16_denoised_Sample1")
    files.append(path + "Pos3To5_lowAngle_16x16_denoised_Sample1")
    path = "/home/kavits/Project/SpinVision/SpinVision/resources/NetworkWeights/test/"
    #files.append(path + "testUntrainedGaussian_1024x40")
    # this is not gonna plot, just there to see if an error is raised

    print "neuronType: " + str(n.__neuronType__)
    print "STDPParameters: " + str(n.__STDPParameters__)
    print "Neuron Params: " + str(n.__neuronParameters__)

    weights = control.train(60, 2, 200, files[0], files[0], plot=True)#, weightSource=files[2])
    print weights

trainWithWeightSource()