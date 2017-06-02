import SpinVision.AEDAT_Handler as handler
import SpinVision.neuralNet as n
import SpinVision.networkControl as control
import numpy
numpy.set_printoptions(threshold=numpy.nan)
basepath = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/"

def truncateRecording(source, start,end, dest):
    fromTime_us = 116896935
    toTime_us = 118825756

    nicedata = handler.truncate(source, start,end,dest)



#truncateRecording(basepath + "Pos3To5_lowAngle", 374545004, 377171417, basepath + "Pos3To5_lowAngle_Sample1")
# truncateRecording(basepath + "Pos3To5_lowAngle_32x32", 381759446, 385325434, basepath + "/TrainingSamples/Pos3To5_lowAngle_32x32/Sample2")
# truncateRecording(basepath + "Pos3To5_lowAngle_32x32", 389753227, 391839810, basepath + "/TrainingSamples/Pos3To5_lowAngle_32x32/Sample3")

traininDirs = []
traininDirs.append(basepath + "Pos3To5_lowAngle_Sample1")
destFile = basepath + "Pos3To5_lowAngle_Sample1_filtered"

source = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/"

handler.readData(source + "10xtestSampleLeft")

# handler.modeDownsample(traininDirs[0], destFile)

# handler.downsample(traininDirs[1], destFile, 8)
# handler.concatenate(traininDirs, 100000, filter="Sample", dest=destFile, save=False)
# handler.speedUp(10, traininDirs[0] + "/Sample1", traininDirs[0] + "/10xSample1")
# handler.speedUp(10, traininDirs[1] + "/Sample1", traininDirs[1] + "/10xSample1")

# connections = n.createGaussianConnections(1024, 40, 0.5, 0.15)
# weights = [[] * 40] * 1024
#
# for i in range(1024):
#     weights[i] = [connections[40*i + j][2] for j in range(40)]
#
# basepath = "/home/kavits/Project/SpinVision/SpinVision/resources/NetworkWeights/"
# destFile = basepath + "1024x40_Gaussian"
#
# control.saveWeights(weights, destFile)