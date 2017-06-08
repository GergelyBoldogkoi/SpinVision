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
    #
    # , 116915910, 118805148, basepath + "Pos3-1_Sample1")
    #  124541359, 126442203, basepath + "Pos3-1_Sample2")
    #  132288534, 134129110, basepath + "Pos3-1_Sample3")
    #
    # 374545004, 377171417, basepath + "Pos3-5_Sample1")
    # 381759446, 385325434, basepath + "Pos3-5_Sample2")
    #  389753227, 391839810, basepath +"Pos3-5_Sample3")


# truncateRecording(basepath + "Pos1-1", 192126147, 194206124, basepath + "Pos1-1_Sample1")
# truncateRecording(basepath + "Pos1-1", 194436240, 196275918, basepath + "Pos1-1_Sample2")
# truncateRecording(basepath + "Pos1-1", 196284571, 198508865, basepath + "Pos1-1_Sample3")
#
# truncateRecording(basepath + "Pos5-5", 268816788, 270656797, basepath + "Pos5-5_Sample1")
# truncateRecording(basepath + "Pos5-5", 270955460, 272795535, basepath + "Pos5-5_Sample2")
# truncateRecording(basepath + "Pos5-5", 273136841, 274977143, basepath + "Pos5-5_Sample3")

# traininDirs = []
# traininDirs.append(basepath + "TrainingSamples/16x16/")
# destFile = basepath + "Pos3To1_lowAngle_16x16_denoised"
#
# #
# path = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/"
# handler.downsample(path + "Pos1-1_Sample1_denoised", path + "Pos1-1_Sample1_denoised_32x32", 4)

# handler.speedUp(2, traininDirs[0] + "Pos3To5_lowAngle_16x16_denoised_Sample1", traininDirs[0] + "/2xPos3To5_lowAngle_16x16_denoised_Sample1")
# handler.speedUp(2, traininDirs[0] + "Pos3To1_lowAngle_16x16_denoised_Sample1", traininDirs[0] + "/2xPos3To1_lowAngle_16x16_denoised_Sample1")
#
# # connections = n.createGaussianConnections(1024, 40, 0.5, 0.15)
# weights = [[] * 40] * 1024
#
# for i in range(1024):
#     weights[i] = [connections[40*i + j][2] for j in range(40)]
#
# basepath = "/home/kavits/Project/SpinVision/SpinVision/resources/NetworkWeights/"
# destFile = basepath + "1024x40_Gaussian"
#
# control.saveWeights(weights, destFile)
# #? basepath + "Pos5-5_Sample2_denoised")