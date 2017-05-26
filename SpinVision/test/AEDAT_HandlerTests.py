import unittest
import SpinVision.AEDAT_Handler as handler
import os

class AEDATHandlerTests(unittest.TestCase):
    __basePath__ = "/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/"
    def canDownsampleAEDAT(self):
        sourceFile = "testRecording"
        destFile = "downsampledTestRecoding"

        handler.downsampleMatlab(sourceFile, destFile,4)

        file = open(self.__basePath__ + destFile + ".aedat", "r")

        file.close()
        os.remove(self.__basePath__ + destFile + ".aedat")

        file = open(self.__basePath__ + destFile + ".mat", "r")

        file.close()
        os.remove(self.__basePath__ + destFile + ".mat")

