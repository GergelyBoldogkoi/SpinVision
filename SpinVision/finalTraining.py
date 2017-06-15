import AEDAT_Handler as f
import neuralNet as n
import pyNN.spiNNaker as p
import random as r
import networkControl as control
import math

path = "/home/kavits/Project/good recording/denoised_32x32/"
files = []

# files.append(path + "1-2_denoised_32x32")

# files.append(path + "1-3_denoised_32x32")
# files.append(path + "1-4_denoised_32x323")

#
# files.append(path + "2-1_denoised_32x32")
# files.append(path + "2-2_denoised_32x32")
# files.append(path + "2-3_denoised_32x32")
# files.append(path + "2-4_denoised_32x32")
# files.append(path + "2-5_denoised_32x32")


# files.append(path + "3-2_denoised_32x32")

# files.append(path + "3-4_denoised_32x32")


# files.append(path + "4-1_denoised_32x32")
# files.append(path + "4-2_ddenoised_32x32")
# files.append(path + "4-3_denoised_32x32")
# files.append(path + "4-4_denoised_32x32")
# files.append(path + "4-5_denoised_32x32")


# files.append(path + "5-2_denoised_32x32")

# files.append(path + "5-4_denoised_32x32")


files.append(path + "3-1_denoised_32x32")
files.append(path + "3-3_denoised_32x32")
files.append(path + "3-5_denoised_32x32")

# files.append(path + "rightOf5-1_denoised_32x32")
files.append(path + "leftOf5-3_denoised_32x32")
files.append(path + "5-5_denoised_32x32")

files.append(path + "1-1_denoised_32x32")
files.append(path + "1-3checkers_denoised_32x32")
# files.append(path + "leftOf1-5_denoised_32x32")

# files.append(path + "5-1_denoised_32x32")

# files.append(path + "1-5_denoised_32x32")




tpPairings = {}

tpPairings[files[0]] = 1
tpPairings[files[1]] = 3
tpPairings[files[2]] = 5

tpPairings[files[3]] = 3
tpPairings[files[4]] = 5

tpPairings[files[5]] = 1
tpPairings[files[6]] = 3



print "tpPairings"
print tpPairings

path2 = "/home/kavits/Project/SpinVision/SpinVision/resources/NetworkWeights/withGoodRecordings/"
# file "trials" has 20 iterations and recognises 7 traj and 5 pos recognises trajectories with huge delay and connstr = 15, inhibstr = 1000


sources = files

files2 = []
files2.append(path + "1-1_denoised_32x32")
# files2.append(path + "1-2_denoised_32x32")
# files2.append(path + "1-3_denoised_32x32")
files2.append(path + "1-3checkers_denoised_32x32")
# files2.append(path + "leftOf1-5_denoised_32x32")
# files2.append(path + "1-4_denoised_32x323")
# files2.append(path + "1-5_denoised_32x32")
#
# files2.append(path + "2-1_denoised_32x32")
# files2.append(path + "2-2_denoised_32x32")
# files2.append(path + "2-3_denoised_32x32")
# files2.append(path + "2-4_denoised_32x32")
# files2.append(path + "2-5_denoised_32x32")

files2.append(path + "3-1_denoised_32x32")
# files2.append(path + "3-2_denoised_32x32")
files2.append(path + "3-3_denoised_32x32")
# files2.append(path + "3-4_denoised_32x32")
files2.append(path + "3-5_denoised_32x32")

# files2.append(path + "4-1_denoised_32x32")
# files2.append(path + "4-2_ddenoised_32x32")
# files2.append(path + "4-3_denoised_32x32")
# files2.append(path + "4-4_denoised_32x32")
# files2.append(path + "4-5_denoised_32x32")

# files2.append(path + "5-1_denoised_32x32")
# files2.append(path + "5-2_denoised_32x32")
# files2.append(path + "rightOf5-1_denoised_32x32")
files2.append(path + "leftOf5-3_denoised_32x32")
# files2.append(path + "5-4_denoised_32x32")
files2.append(path + "5-5_denoised_32x32")

evalSources = files2

saveFiles = []
saveFiles.append(path2 + "20iterNewRecordins")


def trainWithWeightSource():
    # this is not gonna plot, just there to see if an error is raised

    print "neuronType: " + str(n.__neuronType__)
    print "STDPParameters: " + str(n.__STDPParameters__)
    print "Neuron Params: " + str(n.__neuronParameters__)

    weights = control.train_Trajectories(1024, 20, 20, sources, plot=True, save=True, destFile=saveFiles[0])
    # weights = control.trainTrajectories(60, 3, 100, files[0], files[1], plot=True)#, weightSource=files[2])
    # print weights


evalFiles = []
path2 = "/home/kavits/Project/SpinVision/SpinVision/resources/NetworkWeights/withGoodRecordings/"

# file "trials" has 20 iterations and recognises 7 traj and 5 pos recognises trajectories with huge delay and connstr = 15, inhibstr = 1000
evalFiles.append(saveFiles[0])


def trainForEndPos():
    weights = control.loadWeights(evalFiles[0])
    nrPos = 3
    obj = control.train_endPositions(nrPos, sources, weights, tpPairings)
    print obj['pairings']

    hallelujah = control.evaluateEndPositions(1024, evalSources, inputWeights=weights, trainedNetwork=obj['net'])


trainWithWeightSource()
trainForEndPos()
