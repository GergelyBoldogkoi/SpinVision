import paer


def downsampleMatlab(sourceFile, destFile, scale):
    basePath = "/home/kavits/Project/DVS Recordings/Matlab Recordings/"

    sourcePath = basePath + sourceFile + ".aedat"
    lib = paer.aefile(sourcePath)
    data = paer.aedata(lib)

    sampled = data.downsample((int(128/scale), int(128/scale)))

    destPath = basePath + destFile

    lib.save(sampled, destPath + '.aedat', 'aedat') #This file will be viewable in jAER
    sampled.save_to_mat(destPath + '.mat') #This file can be read by the application to extract spiketimes!

def convertToSpikes(filePath):
    return

