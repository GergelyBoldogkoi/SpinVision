import SpinVision.AEDAT_Handler as handler
import SpinVision.neuralNet
import numpy
numpy.set_printoptions(threshold=numpy.nan)
basepath = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/"

def truncateRecording(source, start,end, dest):
    fromTime_us = 116896935
    toTime_us = 118825756

    nicedata = handler.truncate(source, start,end,dest)



# truncateRecording(basepath + "Pos3To5_lowAngle_32x32", 374545004, 377171417, basepath + "/TrainingSamples/Pos3To5_lowAngle_32x32/Sample1")
# truncateRecording(basepath + "Pos3To5_lowAngle_32x32", 381759446, 385325434, basepath + "/TrainingSamples/Pos3To5_lowAngle_32x32/Sample2")
# truncateRecording(basepath + "Pos3To5_lowAngle_32x32", 389753227, 391839810, basepath + "/TrainingSamples/Pos3To5_lowAngle_32x32/Sample3")

traininDirs = []
traininDirs.append(basepath + "TrainingSamples/Pos3To1_lowAngle_32x32")
traininDirs.append(basepath + "TrainingSamples/Pos3To5_lowAngle_32x32")
destFile = basepath + "TrainingSamples/concat15"
# handler.concatenate(traininDirs, 100000, filter="Sample", dest=destFile, save=False)
handler.speedUp(10, traininDirs[0] + "/Sample1", traininDirs[0] + "/10xSample1")
handler.speedUp(10, traininDirs[1] + "/Sample1", traininDirs[1] + "/10xSample1")