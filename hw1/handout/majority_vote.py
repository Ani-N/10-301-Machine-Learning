import sys
import csv
import numpy as np


train_input_path = sys.argv[1]
test_input_path = sys.argv[2]
train_output_path = sys.argv[3]
test_output_path = sys.argv[4]
metrics_output_path = sys.argv[5]

def trainModel():
    numZeros = 0
    numOnes = 0
    majorityVote = 1
    
    trainArr = np.genfromtxt(train_input_path, delimiter="\t")

    numLines = trainArr.shape[0]
    numCols = trainArr.shape[1]
    for line in range(1, (numLines)):
        if trainArr[line, numCols-1] == 1:
            numOnes+=1
        else:
            numZeros+=1
        #print(trainArr[line, numCols-1])
    #print(numOnes)
    #print(numZeros)
    #if numZeros == (numLines-(numOnes+1)):
    #    print("this is redundant!!")
    if numZeros > numOnes:
        majorityVote = 0
    return (numZeros, numOnes, majorityVote)

def testModel(majorityVote):
    numZeros = 0
    numOnes = 0
    errRate = 0

    testArr = np.genfromtxt(test_input_path, delimiter="\t")


    numLines = testArr.shape[0]
    numCols = testArr.shape[1]
    for line in range(1, (numLines)):
        if testArr[line, numCols-1] == 1:
            numOnes+=1
        else:
            numZeros+=1
        #print(testArr[line, numCols-1])
    if majorityVote ==1:
        errRate = numZeros/(numZeros+numOnes)
    else:
        errRate = numOnes/(numZeros+numOnes)
    return (errRate)


def writeError(trainStats, testStats):
    trainError = 0

    #if majority vote = 1, then: error = num0s/total
    if trainStats[2] == 1:
        trainError = trainStats[0]/(trainStats[0]+trainStats[1])
    else:
        trainError = trainStats[1]/(trainStats[0]+trainStats[1])

    #print(trainError)
    #print(testStats)

    with open(metrics_output_path, 'w') as file:
        file.write('error(train): '+ str(trainError) + '\nerror(test): ' + str(testStats))
    file.close

def writeTrainPredictions(majorityVote):
    trainArr = np.genfromtxt(train_input_path, delimiter="\t")
    numLines = (trainArr.shape[0]-1)
    with open(train_output_path, 'w') as file:
        for x in range(numLines):
            file.write(str(majorityVote) + '\n')
    file.close

def writeTestPredictions(majorityVote):
    testArr = np.genfromtxt(test_input_path, delimiter="\t")
    numLines = (testArr.shape[0]-1)
    with open(test_output_path, 'w') as file:
        for x in range(numLines):
            file.write(str(majorityVote) + '\n')
    file.close


#print(trainModel())
training_stats = trainModel()  
test_stats = testModel(training_stats[2])
writeError(training_stats, test_stats)
writeTrainPredictions(training_stats[2])
writeTestPredictions(training_stats[2])

#with np.printoptions(threshold=np.inf):
#    print(heartArr)