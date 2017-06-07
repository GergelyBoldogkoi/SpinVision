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


# truncateRecording(basepath + "Pos3To1_lowAngle_16x16_denoised", 116915910, 118805148, basepath + "TrainingSamples/16x16/Pos3To1_lowAngle_16x16_denoised_Sample1")
# truncateRecording(basepath + "Pos3To1_lowAngle_16x16_denoised", 124541359, 126442203, basepath + "TrainingSamples/16x16/Pos3To1_lowAngle_16x16_denoised_Sample2")
# truncateRecording(basepath + "Pos3To1_lowAngle_16x16_denoised", 132288534, 134129110, basepath + "TrainingSamples/16x16/Pos3To1_lowAngle_16x16_denoised_Sample3")

# truncateRecording(basepath + "Pos3To5_lowAngle_16x16_denoised", 374545004, 377171417, basepath + "TrainingSamples/16x16/Pos3To5_lowAngle_16x16_denoised_Sample1")
# truncateRecording(basepath + "Pos3To5_lowAngle_16x16_denoised", 381759446, 385325434, basepath + "TrainingSamples/16x16/Pos3To5_lowAngle_16x16_denoised_Sample2")
# truncateRecording(basepath + "Pos3To5_lowAngle_16x16_denoised", 389753227, 391839810, basepath + "TrainingSamples/16x16/Pos3To5_lowAngle_16x16_denoised_Sample3")

# traininDirs = []
# traininDirs.append(basepath + "Pos3To1_lowAngle_denoised")
# destFile = basepath + "Pos3To1_lowAngle_16x16_denoised"
#
#
# handler.downsample(traininDirs[0], destFile, 8)

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
#
# data  = handler.readData(basepath + "Pos3To1_lowAngle")
# data  = handler.filterNoise(data,2,8)
# handler.saveData(data, basepath + "Pos3To1_lowAngle_denoised")