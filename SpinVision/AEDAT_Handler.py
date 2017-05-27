import paer
import scipy.io
from os import listdir
import numpy as np



def read(filename):
    sourceFile = filename + ".aedat"
    aer = paer.aefile(sourceFile)
    data = paer.aedata(aer)

    return {'data': data, 'aer': aer}

def readData(filename):
    sourceFile = filename + ".aedat"
    aer = paer.aefile(sourceFile)
    data = paer.aedata(aer)

    return data


def readFile(filename):
    sourceFile = filename + ".aedat"
    aer = paer.aefile(sourceFile)

    return aer

#returns paer.aerdata in a legible format
def extractData(data):
    niceData = {'X': data.x, 'Y': data.y, 't': data.t, 'ts': data.ts}
    return niceData

def downsample(sourceFile, destFile, scale):
    sourcePath = sourceFile + ".aedat"
    lib = paer.aefile(sourcePath)
    data = paer.aedata(lib)

    sampled = data.downsample((int(128 / scale), int(128 / scale)))

    destPath = destFile

    lib.save(sampled, destPath + '.aedat', 'aedat')  # This file will be viewable in jAER
    # sampled.save_to_mat(destPath + '.mat') #This file can be read by the application to extract spiketimes!


def truncate(sourcePath, from_us, to_us, destPath):
    loaded = read(sourcePath)

    data = loaded['data']
    niceData = extractData(data)
    file = loaded['aer']


    prevTs = 0
    includeFrom = len(niceData['ts'])  # ensure nothing gets included without going through loop
    stopAt = 0
    includeIsSet = False

    for i in range(len(niceData['ts'])):
        ts = niceData['ts'][i]

        if ts >= from_us and prevTs <= from_us:
            if not includeIsSet:
                includeFrom = i
                includeIsSet = True

        if ts > to_us and prevTs <= to_us:  # includes events at fromMs and toMs
            stopAt = i
            break

        prevTs = ts

    if stopAt == 0:
        raise RuntimeError("Could not find where to stop")

    data.x = data.x[includeFrom : stopAt]
    data.y = data.y[includeFrom : stopAt]
    data.t = data.t[includeFrom : stopAt]
    data.ts = data.ts[includeFrom : stopAt]


    file.save(data, destPath + ".aedat")
    return extractData(data)
#
def append(sourceDir, timeBetweenSamples_us, dest=None, save=False):
    aeFile = paer.aefile("/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/justAfiletoMakePaerWork.aedat")
    collectiveData = paer.aedata()

    for dir in sourceDir:
        for file in listdir(dir):

            filePath = dir + "/" + file[0:len(file) - len(".aedat")]
            data = readData(filePath)
            collectiveData.x = np.array(collectiveData.x.tolist() + data.x.tolist())
            collectiveData.y = np.array(collectiveData.y.tolist() + data.y.tolist())
            collectiveData.t = np.array(collectiveData.t.tolist() + data.t.tolist())

            indexOfLast = len(collectiveData.ts) - 1
            if indexOfLast < 0:
                collectiveData.ts = data.ts
            else:
                endOfPrev = collectiveData.ts[indexOfLast]
                templist = [ts + endOfPrev + timeBetweenSamples_us for ts in data.ts]
                collectiveData.ts = np.array(collectiveData.ts.tolist() + templist) #templist needed as numpy is just great!

            assert len(collectiveData.ts) == indexOfLast + 1 + len(data.ts)

    if save:
        if dest == None:
            raise AttributeError("Please specify a destination file")

        aeFile.save(collectiveData, dest + ".aedat")
    return collectiveData
