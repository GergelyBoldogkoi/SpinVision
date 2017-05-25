import unittest
import AEDAT_Handler as handler
import os

class AEDAT_HandlerTests(unittest.TestCase):
    __basePath__ = "/home/kavits/Project/DVS Recordings/Matlab Recordings/"
    def canDownsampleAEDAT(self):
        sourceFile = "testRecording"
        destFile = "downsampledTestRecoding"

        handler.downsampleMatlab(sourceFile, destFile,4)

        file = open(self.__basePath__ + sourceFile + ".aedat", "r")

        file.close()
        os.remove(self.__basePath__ + destFile + ".aedat")
