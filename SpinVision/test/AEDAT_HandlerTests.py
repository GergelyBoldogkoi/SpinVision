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

    def test_canRead(self):
        filename = "downsampledTestRecording"
        data = handler.read(self.__basePath__ + filename)

        assert 4 == len(data.keys())
        assert 31632 == len(data['X'])
        assert len(data['X']) == len(data['Y'])
        assert len(data['X']) == len(data['t'])
        assert len(data['X']) == len(data['ts'])

    # def test_canTruncate(self):
        #todo complete
    #     fileName = "downsampledTestRecording"
    #     destFile = "to_del_testTruncatedTestRec"
    #     fromTime_ms = 116896935 / 1000
    #     toTime_ms = 118825756 / 1000
    #
    #     data = handler.truncate(self.__basePath__ + fileName, fromTime_ms, toTime_ms, self.__basePath__ + destFile)
    #
    #     print len(data['ts'])
    #
    #     os.remove(self.__basePath__ + destFile + ".mat")
