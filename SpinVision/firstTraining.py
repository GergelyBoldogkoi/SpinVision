import AEDAT_Handler as f
import neuralNet as n
import pyNN.spiNNaker as p
import random as r
import networkControl as control
import math
#
# path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/TrainingSamples/"
# files = []
# files.append(path + "Pos1-1_Sample1_denoised_32x32")  # NR:1 #These WORK!!
# files.append(path + "Pos5-5_Sample2_denoised_32x32")  # NR:2
# files.append(path + "Pos3-1_Sample3_denoised_32x32")  # NR:3
# files.append(path + "Pos3-5_Sample1_denoised_32x32")  # NR:4
#




path = "/home/kavits/Project/New Recodrings/32x32_denoised_Samples/"
files = []

files.append(path + "1-1_denoised_32x32_Sample2")
# files.append(path + "1-2_denoised_32x32_Sample2")
# files.append(path + "1-3_denoised_32x32_Sample2")
# files.append(path + "1-4_denoised_32x32_Sample3")
# files.append(path + "1-5_denoised_32x32_Sample2")

# files.append(path + "2-1_denoised_32x32_Sample2")
files.append(path + "2-2_denoised_32x32_Sample2")
# files.append(path + "2-3_denoised_32x32_Sample1")
# files.append(path + "2-4_denoised_32x32_Sample1")
# files.append(path + "2-5_denoised_32x32_Sample1")

# files.append(path + "3-1_denoised_32x32_Sample2")
# files.append(path + "3-2_denoised_32x32_Sample2")
files.append(path + "3-3_denoised_32x32_Sample2")
# files.append(path + "3-4_denoised_32x32_Sample2")
# files.append(path + "3-5_denoised_32x32_Sample2")

# files.append(path + "4-1_denoised_32x32_Sample2")
# files.append(path + "4-2_denoised_32x32_Sample1")
# files.append(path + "4-3_denoised_32x32_Sample2")
files.append(path + "4-4_denoised_32x32_Sample2")
# files.append(path + "4-5_denoised_32x32_Sample2")

# files.append(path + "5-1_denoised_32x32_Sample1")
# files.append(path + "5-2_denoised_32x32_Sample1")
# files.append(path + "5-3_denoised_32x32_Sample1")
# files.append(path + "5-4_denoised_32x32_Sample2")
files.append(path + "5-5_denoised_32x32_Sample2")


tpPairings = {}
tpPairings[files[0]] = 1
tpPairings[files[1]] = 2
tpPairings[files[2]] = 3
tpPairings[files[3]] = 4
tpPairings[files[4]] = 5

files2 = []
path2 = "/home/kavits/Project/SpinVision/SpinVision/resources/NetworkWeights/"
files2.append(path2 + "trials")

sources = files

tpPairings = {}
print "files length: " + str(len(files))
poscount = 1
for i in range(0, len(files)):
    tpPairings[files[i]] = poscount
    poscount += 1
    if poscount == 6:
        poscount = 1

print tpPairings


def trainWithWeightSource():
    # this is not gonna plot, just there to see if an error is raised

    print "neuronType: " + str(n.__neuronType__)
    print "STDPParameters: " + str(n.__STDPParameters__)
    print "Neuron Params: " + str(n.__neuronParameters__)

    weights = control.train_Trajectories(1024, 20, 20, sources, plot=True, save=True, destFile=files2[0])
    # weights = control.trainTrajectories(60, 3, 100, files[0], files[1], plot=True)#, weightSource=files[2])
    # print weights


def trainForEndPos():
    weights = control.loadWeights(files2[0])
    obj = control.train_endPositions(5, sources, weights, tpPairings)
    print obj['pairings']
    hallelujah = control.evaluateEndPositions(1024, sources, inputWeights=weights, trainedNetwork=obj['net'])


trainWithWeightSource()
trainForEndPos()
