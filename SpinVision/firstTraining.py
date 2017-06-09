import AEDAT_Handler as f
import neuralNet as n
import pyNN.spiNNaker as p
import random as r
import networkControl as control

path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/TrainingSamples/"
files = []
files.append(path + "Pos1-1_Sample1_denoised_32x32") #These WORK!!
files.append(path + "Pos5-5_Sample2_denoised_32x32")
path2 = "/home/kavits/Project/SpinVision/SpinVision/resources/NetworkWeights/"
files.append(path2 + "1024x10_20iter_2traj")

def trainWithWeightSource():


    # this is not gonna plot, just there to see if an error is raised

    print "neuronType: " + str(n.__neuronType__)
    print "STDPParameters: " + str(n.__STDPParameters__)
    print "Neuron Params: " + str(n.__neuronParameters__)

    weights = control.train_Trajectories(1024, 10, 50, [files[0], files[1]], plot=True, save=True, destFile=files[2])
    # weights = control.trainTrajectories(60, 3, 100, files[0], files[1], plot=True)#, weightSource=files[2])
    # print weights

def trainForEndPos():
    sources = [files[0], files[1]]

    weights = control.loadWeights(files[2])
    net = control.train_endPositions(2, sources, weights)
    hallelujah = control.evaluateEndPositions(1024, sources, inputWeights=weights, trainedNetwork=net)


# trainWithWeightSource()
trainForEndPos()
