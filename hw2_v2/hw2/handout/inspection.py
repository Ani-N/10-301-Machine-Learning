#entropy of classifying WHEN USING A MAJORITY VOTE
#use majority vote code to extract info on # of 0/1, then use entropy formula using log base 2

import sys
import numpy as np
import math

#assigning arguments to variables
input_path = sys.argv[1]
output_path = sys.argv[2]

data_arr = np.genfromtxt(input_path, delimiter="\t")
sliced_arr = data_arr[1:, -1:]

def Plog2P(probability):
    if (probability == 0):
        return 0.0
    return(probability * math.log2(probability))
    
def label_entropy(count1, count2):
    P1 = count1/(count1+count2)
    P2 = count2/(count1+count2)
    return -(Plog2P(P1)+Plog2P(P2))

def inspect():
    last_col_summary = np.unique(sliced_arr, return_counts=True)
    error_rate = 1- (last_col_summary[1].max()/(np.sum(last_col_summary[1])))
    entropy = label_entropy(last_col_summary[1][0], last_col_summary[1][1])
    return((entropy, error_rate))

def writeFile(entropy, inspect_error):
    with open(output_path, 'w') as file:
        file.write('entropy: '+ str(entropy) + '\nerror: ' + str(inspect_error))
    file.close

#print(sorted(np.unique(sliced_arr, return_counts=True)))
entropy, inspect_error = inspect()
writeFile(entropy, inspect_error)