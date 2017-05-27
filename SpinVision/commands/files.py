import SpinVision.AEDAT_Handler as handler
import SpinVision.neuralNet

basepath = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/"

def truncateRecording(source, start,end, dest):
    fromTime_us = 116896935
    toTime_us = 118825756

    nicedata = handler.truncate(source, start,end,dest)

# truncateRecording(basepath + "Pos3To1_lowAngle_32x32", 124523251, 126470965, basepath + "/TrainingSamples/Pos3To1_lowAngle_32x32/Sample2")
# truncateRecording(basepath + "Pos3To1_lowAngle_32x32", 132261051, 134224138, basepath + "/TrainingSamples/Pos3To1_lowAngle_32x32/Sample3")
