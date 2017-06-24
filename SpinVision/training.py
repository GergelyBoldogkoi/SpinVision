import AEDAT_Handler as f
import neuralNet as n
import pyNN.spiNNaker as p
import random as r
import networkControl as control
import math

filepath = __file__[0:len(__file__) - 11]

path = filepath + "resources/DVS Recordings/denoised_32x32/"
path2 = filepath + "resources/DVS Recordings/veryDenoised_32x32/"
files = []

files.append(path2 + "3-1_denoised_32x32")

files.append(path2 + "3-3_denoised_32x32")

files.append(path2 + "3-5_denoised_32x32")

files.append(path2 + "1-1_denoised_32x32")

files.append(path2 + "1-3checkers_denoised_32x32")


files.append(path2 + "leftOf5-3_denoised_32x32")

files.append(path2 + "5-5_denoised_32x32")

files.append(path2 + "1-5_denoised_32x32")

files.append(path2 + "5-1_denoised_32x32")

sources = files



tpPairings = {}

tpPairings[files[0]] = 1

tpPairings[files[1]] = 3
tpPairings[files[2]] = 5

tpPairings[files[3]] = 1
tpPairings[files[4]] = 3


tpPairings[files[5]] = 3

tpPairings[files[6]] = 5

tpPairings[files[7]] = 5
tpPairings[files[8]] = 1



print "tpPairings"
print tpPairings

path2 = filepath + "resources/NetworkWeights/withGoodRecordings/"


#FROM FAR RIGHT
path = filepath + "resources/DVS Recordings/denoised_32x32/"
files2 = []
files2.append(path + "farRightOf5-5_denoised_32x32")
files2.append(path + "fromFarRightToFarLeft1-3_denoised_32x32")
files2.append(path + "fromFarRightToFarLeft1-5_denoised_32x32")

files2.append(path + "fromFarRightToFarLeft3-1_denoised_32x32")
files2.append(path + "fromFarRightToFarLeft3-5_denoised_32x32")
files2.append(path + "fromFarRightToFarRight1-3_denoised_32x32")
files2.append(path + "fromFarRightToFarRight3-1_denoised_32x32")
files2.append(path + "fromFarRightToFarRight3-5_denoised_32x32")


evalSources = files2

saveFiles = []
saveFiles.append(path2 + "20iterNewRecordins")


def trainWithWeightSource():

    print "neuronType: " + str(n.__neuronType__)
    print "STDPParameters: " + str(n.__STDPParameters__)
    print "Neuron Params: " + str(n.__neuronParameters__)

    weights = control.train_Trajectories(1024, 20, 5, sources, plot=True, save=True, destFile=saveFiles[0])

evalFiles = []
path2 = filepath + "resources/NetworkWeights/Final Implementation/weights"


evalFiles.append(path2)#"/home/kavits/Pictures/evaluation/Trained with noisy and ot noisy recordings/weights")
def trainForEndPos():
    weights = control.loadWeights(evalFiles[0])
    nrPos = 3
    obj = control.train_endPositions(nrPos, sources, weights, tpPairings)
    print obj['pairings']

    spikes = control.evaluateEndPositions(1024, evalSources, inputWeights=weights, trainedNetwork=obj['net'])


trainWithWeightSource()
trainForEndPos()
