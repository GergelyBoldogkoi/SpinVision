import unittest
import SpinVision.AEDAT_Handler as handler
import os
import numpy


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

        data = handler.append([sourceDir + "/appendTest"], 1000000, sourceDir + destFile, True)
        file = open(self.__basePath__ + destFile + ".aedat", "r")

        file.close()

        os.remove(sourceDir + "/to_del_testAppended.aedat")


