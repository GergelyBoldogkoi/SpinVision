import paer
import scipy.io


# ----------------------------------------------------------------------------
#           THE FOLLOWING FUNCTION HAS BEEN TAKEN FROM HANYI HUs FYP
#               AND HAS BEEN MODIFIED
# ----------------------------------------------------------------------------
def read(filename):
    sourceFile = filename + ".aedat"
    aer = paer.aefile(sourceFile)
    data = paer.aedata(aer)

    niceData = {'X': data.x, 'Y': data.y, 't': data.t, 'ts': data.ts}
    return niceData


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def downsample(sourceFile, destFile, scale):
    sourcePath = sourceFile + ".aedat"
    lib = paer.aefile(sourcePath)
    data = paer.aedata(lib)

    sampled = data.downsample((int(128 / scale), int(128 / scale)))

    destPath = destFile

    lib.save(sampled, destPath + '.aedat', 'aedat')  # This file will be viewable in jAER
    # sampled.save_to_mat(destPath + '.mat') #This file can be read by the application to extract spiketimes!


# def truncate(sourcePath, fromUs, toUs, destPath):
#todo complete
#     data = read(sourcePath)
#
#     print "Length of timestamps" + len(data[3])
#     prevTs = 0
#     includeFrom = len(data[3])  # ensure nothing gets included without going through loop
#     stopAt = 0
#     includeIsSet = False
#     for i in range(len(data[3])):
#         ts = data[3][i]
#         if ts >= fromUs and prevTs <= fromUs:
#             if not includeIsSet:
#                 print "found start at " + str(ts)
#                 includeFrom = i
#                 includeIsSet = True
#
#         if ts > toUs and prevTs <= toUs:  # includes events at fromMs and toMs
#             print "found end at " + str(ts)
#             stopAt = i
#             prevTs = i
#             break
#     if stopAt == 0:
#         raise RuntimeError("Could not find where to stop")
#     truncatedData = {'X': data[0][includeFrom:stopAt],
#                      'Y': data[1][includeFrom:stopAt],
#                      't': data[2][includeFrom:stopAt],
#                      'ts': data[3][includeFrom:stopAt]}
#
#     scipy.io.savemat(destPath + ".mat", truncatedData)
#     return truncatedData
