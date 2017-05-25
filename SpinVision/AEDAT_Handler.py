import paer


def downsampleMatlab(sourceFile, destFile, scale):
    basePath = "/home/kavits/Project/DVS Recordings/Matlab Recordings/"

    sourcePath = basePath + sourceFile + ".aedat"
    lib = paer.aefile(sourcePath)
    data = paer.aedata(lib)

    sampled = data.downsample((int(128/scale), int(128/scale)))

    destPath = basePath + destFile + ".aedat"
    lib.save(sampled, destPath, 'aedat')

