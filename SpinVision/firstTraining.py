import AEDAT_Handler as f
import neuralNet as n
import pyNN.spiNNaker as p
import random as r
import networkControl as control


def trainWithWeightSource():
    path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/"
    files = []
    files.append(path + "10xtestSampleLeft")
    files.append(path + "10xtestSampleRight")
    path = "/home/kavits/Project/SpinVision/SpinVision/resources/NetworkWeights/test/"
    #files.append(path + "testUntrainedGaussian_1024x40")
    # this is not gonna plot, just there to see if an error is raised
    weights = control.train(1024, 40, 100, files[0], files[1], plot=True)#, weightSource=files[2])

trainWithWeightSource()