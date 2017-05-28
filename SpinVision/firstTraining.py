import AEDAT_Handler as f
import neuralNet as n

Network = n.NeuralNet(0.001)
basepath = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/TrainingSamples/"
traininDirs = []
traininDirs.append(basepath + "Pos3To1_lowAngle_32x32")
traininDirs.append(basepath + "Pos3To5_lowAngle_32x32")
destFile = basepath + "concat1_5"

timeBetweenSamples = 100000

inputSpikes = n.getTrainingData(traininDirs, filter="concat", timeBetweenSamples=timeBetweenSamples)
