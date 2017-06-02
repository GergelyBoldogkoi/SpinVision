import paer
import scipy.io
from os import listdir
import numpy as np
np.set_printoptions(threshold=np.nan)


EXPOSURE_TIME = 10

def read(filename):
    sourceFile = filename + ".aedat"
    aer = paer.aefile(sourceFile)
    data = paer.aedata(aer)

    return {'data': data, 'aer': aer}

def readData(filename):
    sourceFile = filename + ".aedat"
    aer = paer.aefile(sourceFile)
    data = paer.aedata(aer)
    # print data.x
    # print type(data.x)

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

def modeDownsample(sourceFile, destFile, scale=4, exposureTime_ms=EXPOSURE_TIME):
    #Todo test
    sourcePath = sourceFile + ".aedat"
    lib = paer.aefile(sourcePath)
    data = paer.aedata(lib)
    newData = filterMode(data, scale, exposureTime_ms)

    lib.save(newData, destFile + '.aedat', 'aedat')


def filterMode(data, scale, exposureTime_ms=50):
    #TODO fix by continous mode calcultion
    print "data.x: " + str(data.x)
    print type(data.x)
    newDim = (int(128 / scale), int(128 / scale))

    assert data.dimensions[0] % newDim[0] is 0
    assert data.dimensions[1] % newDim[1] is 0

    dataByTs = {}
    for i in range(len(data.ts)):  # TODO make faster, could be made faster by reducing time resolution
        # group data by timestamp in order to create "frames"
        key = int(data.ts[i] / (1000 * exposureTime_ms))
        if key not in dataByTs:
            dataByTs[key] = [(data.x[i], data.y[i], data.t[i])]
        else:
            dataByTs[key] += [(data.x[i], data.y[i], data.t[i])]

    x = []
    y = []
    t = []
    tslist = []

    blockNr = 128 / scale  # Number of blocks in the output is given by blockNr * blockNr
    blockSize = scale * scale
    # now iterate though frames
    loopcounter = 0
    for ts in dataByTs:
        print "ts"
        print ts
        spikes = dataByTs[ts]

        # for all frames
        '''For all pixels'''
        for rowBlock in range(blockNr):
            xBlock = rowBlock * scale
            for columnBlock in range(blockNr):
                # for all blocks of size(128/scale)
                nrOff = 0
                nrOn = 0
                yBlock = columnBlock * scale
                for i in range(scale):
                    xIndex = xBlock + i
                    for j in range(scale):
                        # for each pixel in a block
                        yIndex = yBlock + j
                        # count nr of off and on events in block
                        if (xIndex, yIndex, 0) in spikes:
                            nrOff += 1
                        elif (xIndex, yIndex, 1) in spikes:
                            nrOn += 1
                if nrOn > 1:
                    print "nrOn: " + str(nrOn) + " in block " + str(rowBlock) + ", " + str(columnBlock)
                if nrOff > 1:
                    print "nrOff: " + str(nrOff) + " in block " + str(rowBlock) + ", " + str(columnBlock)

                result = None
                # if OFF events are the mode
                if nrOff > nrOn and nrOff >= (blockSize - nrOn) / 2:
                    result = 0
                # if ON events are the mode
                elif nrOn >= nrOff and nrOn >= (blockSize - nrOff) / 2:
                    result = 1

                if result is not None:
                    t += [result]
                    x += [float(rowBlock)]
                    y += [float(columnBlock)]
                    tslist += [float(ts) * 1000 * exposureTime_ms]

    data.x = np.array(x)
    data.y = np.array(y)
    data.t = np.array(t)
    data.ts = np.array(tslist)

    print len(data.x)
    print "newdata.x: " + str(data.x)
    return data


def speedUp(factor, sourceFile, destPath):
    if factor == 0:
        raise ValueError("Cannot speed up by 0")

    sourcePath = sourceFile + ".aedat"
    lib = paer.aefile(sourcePath)
    data = paer.aedata(lib)

    timeStart = data.ts[0]

    fasterData = paer.aedata()
    fasterData.x = data.x
    fasterData.y = data.y
    fasterData.t = data.t

    fasterData.ts = [timeStart + (timeStamp - timeStart)/factor for timeStamp in data.ts]
    lib.save(fasterData, destPath + '.aedat', 'aedat')  # This file will be viewable in jAER

    return fasterData

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
def concatenate(sourceDir, timeBetweenSamples_us, filter=None, dest=None, save=False):
    aeFile = paer.aefile("/home/kavits/Project/SpinVision/SpinVision/resources/DVS Recordings/test/justAfiletoMakePaerWork.aedat")
    collectiveData = paer.aedata()

    for folder in sourceDir:
        for dataFile in listdir(folder):
            if filter is None \
                    or filter == "" \
                    or filter in dataFile:
                print dataFile
                collectiveData = merge(collectiveData, folder, dataFile, timeBetweenSamples_us)
            else:
                print "Ignoring " + dataFile + " for training"

    if save:
        if dest is None:
            raise AttributeError("Please specify a destination file")

        aeFile.save(collectiveData, dest + ".aedat")
    return collectiveData


def merge(collectiveData, dir, file, timeBetweenSamples_us):
    filePath = dir + "/" + file[0:len(file) - len(".aedat")]
    data = readData(filePath)
    firstTs = data.ts[0]

    collectiveData.x = np.array(collectiveData.x.tolist() + data.x.tolist())
    collectiveData.y = np.array(collectiveData.y.tolist() + data.y.tolist())
    collectiveData.t = np.array(collectiveData.t.tolist() + data.t.tolist())
    indexOfLast = len(collectiveData.ts) - 1
    if indexOfLast <= 0:
        collectiveData.ts = data.ts
    else:
        endOfPrev = collectiveData.ts[indexOfLast]
        templist = [ts - firstTs + endOfPrev + timeBetweenSamples_us for ts in data.ts]
        collectiveData.ts = np.array(collectiveData.ts.tolist() + templist)  # templist needed as numpy is just great!
    assert len(collectiveData.ts) == indexOfLast + 1 + len(data.ts)
    return collectiveData
