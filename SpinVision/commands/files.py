import SpinVision.AEDAT_Handler as handler
import SpinVision.neuralNet as n
import SpinVision.networkControl as control
import numpy
numpy.set_printoptions(threshold=numpy.nan)
basepath = "/home/kavits/Project/New Recordings/"
import os

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

def denoise(sources, dests, windowSize, threshold):
    for i in range(len(sources)):
        data = handler.readData(sources[i])
        filtered = handler.filterNoise(data, windowSize, threshold)

        handler.saveData(filtered,dests[i])

def truncateFiles(recordings):
    for trajectory in recordings.keys():
        for sample in range(len(recordings[trajectory])):
            sourcePath = basepath + "32x32_denoisedWithHighThreshold/" + str(trajectory) + "_denoisedWHT_32x32"
            destPath = "/home/kavits/Project/New Recordings/32x32_denoisedWHT_Samples/" + str(trajectory) + "_denoisedWHT_32x32_Sample" + str(sample+1)
            print destPath
            print recordings[trajectory][sample][0], recordings[trajectory][sample][1]
            from_us = int( recordings[trajectory][sample][0])
            to_us = int(recordings[trajectory][sample][1])
            handler.truncate(sourcePath, from_us, to_us, destPath)




def readTruncationParams():
    recordings = {}
    with open(basepath + "Truncation input", 'r') as f:

        for line in f:
            contents = line.split()
            if contents[0] not in recordings.keys():
                recordings[contents[0]] = [[contents[1], contents[2]]]
            else:
                recordings[contents[0]].append([contents[1], contents[2]])

    return recordings

homedir = '/home/kavits/Project/good recording'
def denoiseAll():
    for filename in os.listdir(homedir):
        if 'denoised' in filename or 'Denoised' in filename:
            continue
        print homedir + '/' + filename[0:len(filename) - 6]
        denoise([homedir + '/' + filename[0:len(filename) - 6]], [homedir + '/veryDenoised_32x32/' + filename[0:len(filename) - 6] + '_denoised'], 1, 7)
#


def downsampleAll():
    for filename in os.listdir('/home/kavits/Project/good recording/denoised'):
        print '/home/kavits/Project/good recording/denoised/' + filename[0:len(filename) - 6]
        handler.downsample('/home/kavits/Project/good recording/denoised/' + filename[0:len(filename) - 6],
                           '/home/kavits/Project/good recording/denoised_32x32/' + filename[0:len(filename) - 6] + '_32x32',
                           4)

def speedUpAll():
    for filename in os.listdir('/home/kavits/Project/good recording/denoised_32x32'):
        handler.speedUp(5,'/home/kavits/Project/good recording/denoised_32x32/' + filename[0:len(filename) - 6],
                        '/home/kavits/Project/good recording/denoised_32x32_5x/' + filename[
                                                                                0:len(filename) - 6] + '_5x',)
denoiseAll()
# sources = []
# dests = []
# for i in range(1,6):d
#     for j in range(1,6):
#         sources.append(basepath + "denoisedWithHighThreshold/" + str(i) + "-" + str(j) + "_denoisedWHT")
#         dests.append(basepath + "32x32_denoisedWithHighThreshold/" + str(i) + "-" + str(j) + "_denoisedWHT_32x32")

# denoise(sources,dests, windowSize=3, threshold=10)

# for i in range(len(sources)):
#     handler.downsample(sources[i],dests[i],4)
# sources = []
# dests = []
# for i in range(1,6):
#     for j in range(1,6):
#         sources.append(basepath + "denoised/" + str(i) + "-" + str(j) + "_denoised")
#         dests.append(basepath + "32x32_denoised/" + str(i) + "-" + str(j) + "_denoised_32x32")
#
# print sources
# print dests
#
# for i in range(len(sources)):
#     handler.downsample(sources[i], dests[i], 4)



# truncateRecording(basepath + "Pos1-1", 192126147, 194206124, basepath + "Pos1-1_Sample1")
# truncateRecording(basepath + "Pos1-1", 194436240, 196275918, basepath + "Pos1-1_Sample2")
# truncateRecording(basepath + "Pos1-1", 196284571, 198508865, basepath + "Pos1-1_Sample3")
#

# truncateRecording(basepath + "Pos5-5", 268816788, 270656797, basepath + "Pos5-5_Sample1")
# truncateRecording(basepath + "Pos5-5", 270955460, 272795535, basepath + "Pos5-5_Sample2")
# truncateRecording(basepath + "Pos5-5", 273136841, 274977143, basepath + "Pos5-5_Sample3")

# truncateRecording(basepath + "Pos3To5_lowAngle_32x32_denoised", 374545004, 377171417, basepath + "Pos3-5_32x32_denoised_Sample1")
# truncateRecording(basepath + "Pos3To5_lowAngle_32x32_denoised",381759446, 385325434, basepath + "Pos3-5_32x32_denoised_Sample2")
# truncateRecording(basepath + "Pos3To5_lowAngle_32x32_denoised",389753227, 391839810, basepath +"Pos3-5_32x32_denoised_Sample3")
#
# truncateRecording(basepath + "Pos3To1_lowAngle_32x32_denoised", 116915910, 118805148, basepath + "Pos3-1_32x32_denoised_Sample1")
# truncateRecording(basepath + "Pos3To1_lowAngle_32x32_denoised",124541359, 126442203, basepath + "Pos3-1_32x32_denoised_Sample2")
# truncateRecording(basepath + "Pos3To1_lowAngle_32x32_denoised",132288534, 134129110, basepath + "Pos3-1_32x32_denoised_Sample3")

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


