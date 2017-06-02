import unittest
import SpinVision.AEDAT_Handler as handler
import os
import numpy as np
import paer
import logging


# numpy.set_printoptions(threshold=numpy.nan)

class AEDATHandlerTests(unittest.TestCase):
    __basePath__ = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/"

    def test_canDownsample(self):
        sourceFile = "testRecording"
        destFile = "to_del_downsampledTestRecording"

        handler.downsample(self.__basePath__ + sourceFile, self.__basePath__ + destFile, 4)

        file = open(self.__basePath__ + destFile + ".aedat", "r")

        file.close()
        os.remove(self.__basePath__ + destFile + ".aedat")

        # file = open(self.__basePath__ + destFile + ".mat", "r")
        #
        # file.close()
        # os.remove(self.__basePath__ + destFile + ".mat")

    def test_canReadData(self):
        filename = "downsampledTestRecording"
        data = handler.readData(self.__basePath__ + filename)
        nicedata = handler.extractData(data)

        assert 4 == len(nicedata.keys())
        assert 31632 == len(nicedata['X'])
        assert len(nicedata['X']) == len(nicedata['Y'])
        assert len(nicedata['X']) == len(nicedata['t'])
        assert len(nicedata['X']) == len(nicedata['ts'])

        dic = {}
        for i in range(1,3):
            if (1,2,3) not in dic:
                dic[(1,2,3)] = [i]
            else:
                dic[(1,2,3)].append(i)

        print dic



    def test_canTruncate(self):

        fileName = "downsampledTestRecording"
        destFile = "to_del_testTruncatedTestRec"
        fromTime_us = 116896935
        toTime_us = 118825756

        nicedata = handler.truncate(self.__basePath__ + fileName, fromTime_us, toTime_us, self.__basePath__ + destFile)

        assert 4 == len(nicedata.keys())
        assert 7494 == len(nicedata['X'])
        assert len(nicedata['X']) == len(nicedata['Y'])
        assert len(nicedata['X']) == len(nicedata['t'])
        assert len(nicedata['X']) == len(nicedata['ts'])

        print max(nicedata['X'])
        print max(nicedata['Y'])
        print min(nicedata['X'])
        print min(nicedata['Y'])

        os.remove(self.__basePath__ + destFile + ".aedat")

    def test_canAppend(self):
        sourceDir = self.__basePath__
        destFile = "/to_del_testAppended"

        data = handler.concatenate([sourceDir + "/appendTest"], 1000000, filter=None, dest=sourceDir + destFile, save=True)
        file = open(self.__basePath__ + destFile + ".aedat", "r")

        file.close()

        os.remove(sourceDir + "/to_del_testAppended.aedat")

    def test_canSpeedUp(self):

        sourceFile =self.__basePath__ + "/downsampledTestRecording"
        destFile = "/spedUp"
        speedFactor = 10
        handler.speedUp(speedFactor, sourceFile, self.__basePath__ + destFile)

    @unittest.skip
    def test_canFilterMode(self):

        firstTimestamps = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        secondTimestamps = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

        #first tests if ON events are recognised correctly
        firstX = [0, 0, 0, 1, 2, 2, 3, 3, 16, 17]
        firstY = [4, 5, 7, 6, 5, 7, 5, 6, 16, 17]
        firstT = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # second tests if OFF events are recognised correctly
        secondX = [16, 17, 0, 0, 0, 1, 2, 2, 3, 3]
        secondY = [15, 18, 4, 5, 7, 6, 5, 7, 5, 6]
        secondT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # third tests that the Mode actually wins
        thirdTs = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        thirdX = [0, 0, 0, 1, 2, 2, 3, 3, 3, 1, 1]
        thirdY = [4, 5, 7, 6, 5, 7, 5, 6, 7, 5, 7]
        thirdT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

        # fourth tests that the in case of equal on and off on wins
        fourthTs = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        fourthX = [0, 0, 0, 1, 2, 2, 3, 3, 3, 1, 1, 3]
        fourthY = [4, 5, 7, 6, 5, 7, 5, 6, 7, 5, 7, 4]
        fourthT = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]


        data = paer.aedata()
        data.ts = np.array(firstTimestamps + secondTimestamps + thirdTs + fourthTs)
        data.x = np.array(firstX + secondX + thirdX + fourthX)
        data.y = np.array(firstY + secondY + thirdY + fourthY)
        data.t = np.array(firstT + secondT + thirdT + fourthT)

        data = handler.filterMode(data, 4)

        print data.x.tolist()
        print data.y.tolist()
        print data.t.tolist()
        print data.ts.tolist()

        self.assertEquals([0, 0, 0, 0], data.x.tolist())
        self.assertEquals([1, 1, 1, 1], data.y.tolist())
        self.assertEquals([1, 0, 0, 1], data.t.tolist())
        self.assertEquals([0, 1, 2, 3], data.ts.tolist())

    @unittest.skip
    def test_canMeanDownSample(self):

        source  = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/"
        source2 = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/"

        data = handler.modeDownsample(source2 + "Pos3To5_lowAngle_Sample1", source2 + "whatNow")





