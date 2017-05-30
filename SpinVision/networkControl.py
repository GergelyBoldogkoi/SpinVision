import neuralNet as n
import AEDAT_Handler as f

TIME_BETWEEN_ITERATIONS = 100
ITERATIONS = 5

def trainFromFile(inSize, outSize, source1,source2):
    weights = []
    with n.NeuralNet() as net:
        net.trainFromFile(inSize, outSize, ITERATIONS, TIME_BETWEEN_ITERATIONS, source1, source2)


    return weights