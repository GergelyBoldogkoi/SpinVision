import paer
import scipy.io



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
                print "found start at " + str(ts)
                includeFrom = i
                includeIsSet = True

        if ts > to_us and prevTs <= to_us:  # includes events at fromMs and toMs
            print "found end at " + str(ts)
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
# def append(source1, source2, dest):
#     data1 = readData(source1)
#     data2 = readData(source2)
#
#     aeFile = paer.aefile("")
#
#     data1.x += data2.x
#     data1.y += data2.y
#     data1.t += data2.t
#     data1.ts += data2.ts
#
#     aeFile.save(data1, dest + ".aedat")
#     return extractData(data1)
