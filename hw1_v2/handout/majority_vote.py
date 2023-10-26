import sys
import csv
import numpy as np

'''
Redoing the assignment to account for non-0/1 binary classifications (such as apple/banana, red/blue, etc).
Also making more use of built-in numpy functions and tools

basic Numpy functions and information from datacamp's NumPy cheat-sheet
https://www.datacamp.com/cheat-sheet/numpy-cheat-sheet-data-analysis-in-python

as well as official numpy docs (numpy.org)

'''
#assigning arguments to variables
train_input_path = sys.argv[1]
test_input_path = sys.argv[2]
train_output_path = sys.argv[3]
test_output_path = sys.argv[4]
metrics_output_path = sys.argv[5]


#opens both datasets as numpy arr from .tsv
train_data_arr = np.genfromtxt(train_input_path, delimiter="\t")
test_data_arr = np.genfromtxt(test_input_path, delimiter="\t")
#separates the last column of each dataset, removes label column
train_data_arr_sliced = train_data_arr[1:, -1:]
test_data_arr_sliced = test_data_arr[1:, -1:]

""" print statements
with np.printoptions(threshold=np.inf):
    print(train_data_arr_sliced)

print(type(np.unique(train_data_arr_sliced, return_counts=True)[1][1]))
print(np.unique(train_data_arr_sliced, return_counts=True)) """

def trainModel():
    last_col_summary = np.unique(train_data_arr_sliced, return_counts=True)
    majority_vote = last_col_summary[0][last_col_summary[1].argmax()]
    if last_col_summary[1][0]==last_col_summary[1][1]:
        if last_col_summary[0][0]>last_col_summary[0][1]:
            majority_vote = last_col_summary[0][0]
        else:
            majority_vote = last_col_summary[0][1]
    error_rate = 1- (last_col_summary[1].max()/(np.sum(last_col_summary[1])))
    return((majority_vote, error_rate))


def testModel(majority_vote):
    last_col_summary = np.unique(test_data_arr_sliced, return_counts=True)
    error_rate = 1-(last_col_summary[1][np.where(last_col_summary[0]==majority_vote)])/np.sum(last_col_summary[1])
    return(error_rate[0])

def writeError(train_error, test_error):
    with open(metrics_output_path, 'w') as file:
        file.write('error(train): '+ str(train_error) + '\nerror(test): ' + str(test_error))
    file.close

def writeTrainPredictions(majority_vote):
    with open(train_output_path, 'w') as file:
        for x in range(train_data_arr_sliced.size):
            file.write(str(majority_vote) + '\n')
    file.close

def writeTestPredictions(majority_vote):
    with open(test_output_path, 'w') as file:
        for x in range(test_data_arr_sliced.size):
            file.write(str(majority_vote) + '\n')
    file.close

majority_vote, train_error = trainModel()
test_error = testModel(majority_vote)
writeError(train_error, test_error)
writeTrainPredictions(majority_vote)
writeTestPredictions(majority_vote)



'''
#print(trainModel())
training_stats = trainModel()  
test_stats = testModel(training_stats[2])
writeError(training_stats, test_stats)
writeTrainPredictions(training_stats[2])
writeTestPredictions(training_stats[2])


'''