import argparse
import sys
import numpy as np
import math

class Node:
     def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None
        self.num0 = 0
        self.num1 = 0


def Plog2P(probability):
    if (probability == 0):
        return 0.0
    return(probability * math.log2(probability))
#for H(Y) and H(Y|X=x)
def label_entropy(count1, count2):
    P1 = count1/(count1+count2)
    P2 = count2/(count1+count2)
    return -(Plog2P(P1)+Plog2P(P2))

def get_column(index, input_arr):
    return input_arr[1:, index]

def get_label_column(input_arr):
    return input_arr[1:, -1]

def feature_label_combos(col_index, input_arr):
    feature_0_arr = input_arr[input_arr[:,col_index] == '0']
    f0_y0_arr = feature_0_arr[feature_0_arr[:,-1]=='0']
    f0_y1_arr = feature_0_arr[feature_0_arr[:,-1]=='1']

    feature_1_arr = input_arr[input_arr[:,col_index] == '1']
    f1_y0_arr = feature_1_arr[feature_1_arr[:,-1]=='0']
    f1_y1_arr = feature_1_arr[feature_1_arr[:,-1]=='1']

    #returns all combos of X=x and Y=y for binary feature/label
    return(np.shape(f0_y0_arr)[0], np.shape(f0_y1_arr)[0],
           np.shape(f1_y0_arr)[0],np.shape(f1_y1_arr)[0])

def mutual_info(col_index, input_arr):
    f0_y0, f0_y1, f1_y0, f1_y1 = feature_label_combos(col_index, input_arr)

    P_f0 = ((f0_y0+f0_y1)/(f0_y0 + f0_y1 + f1_y0 + f1_y1))
    P_f1 = (1- P_f0)

    if(P_f0 == 0) or(P_f1 == 0):
        return 0

    entropy_given_f0 = label_entropy(f0_y0, f0_y1)
    entropy_given_f1 = label_entropy(f1_y0, f1_y1)

    conditional_entropy = (P_f0 * entropy_given_f0)+(P_f1 * entropy_given_f1)
    entropy_Y = label_entropy((f0_y0+f1_y0), (f1_y0+f1_y1))

    return (entropy_Y-conditional_entropy)


def determine_split(input_arr):
    info_per_feature = []
    for col_index in range(np.shape(input_arr)[1]-1):
        info_per_feature.append(mutual_info(col_index, input_arr))
    max_info = max(info_per_feature)
    return (info_per_feature.index(max_info), max_info)

def left_half(col_index, input_arr):
    feature_0_arr = input_arr[input_arr[:,col_index] == '0']
    named_half_arr = np.vstack([input_arr[0], feature_0_arr])
    return(named_half_arr)

def right_half(col_index, input_arr):
    feature_1_arr = input_arr[input_arr[:,col_index] == '1']
    named_half_arr = np.vstack([input_arr[0], feature_1_arr])
    return(named_half_arr)

def majority_vote(input_arr):
    y0_arr = input_arr[input_arr[:,-1]=='0']
    y1_arr = input_arr[input_arr[:,-1]=='1']

    y0_len = np.shape(y0_arr)[0]
    y1_len = np.shape(y1_arr)[0]

    if y0_len > y1_len:
        return "0"
    if y1_len >= y0_len:
        return "1"

def train_model(input_arr, depth):
    if(depth >= args.max_depth) or (np.shape(input_arr)[1]==1):
        leaf_node = Node()
        leaf_node.vote = majority_vote(input_arr)
        return leaf_node
    split_index, max_info = determine_split(input_arr)
    if(max_info==0):
        leaf_node = Node()
        leaf_node.vote = majority_vote(input_arr)
        return leaf_node
    
    left_data = left_half(split_index, input_arr)
    right_data = right_half(split_index, input_arr)

    split_node = Node()

    split_node.attr = input_arr[0][split_index]
    split_node.left = train_model(left_data, depth+1)
    split_node.right = train_model(right_data, depth+1)

    print(split_index)
    print(split_node.attr)

    combos = feature_label_combos(split_index, input_arr)
    split_node.num0 = (combos[0]+ combos[2])
    split_node.num1 = (combos[1]+ combos[3])
    return split_node

def evaluate_item(item, tree_node):
    if(tree_node.vote != None):
        #print("result found: "+str(tree_node.vote))
        return tree_node.vote
    
    #print("evaluating "+ tree_node.attr)
    col_index = (np.where(train_data_arr[0] == tree_node.attr)[0][0])

    if (item[col_index] == '0') and (tree_node.left != None):
        #print("item is " + item[col_index]+ ": searching left...")
        return(evaluate_item(item, tree_node.left))
    if (item[col_index] =='1') and (tree_node.right != None):
        #print("item is " + item[col_index]+ ": searching right...")
        return(evaluate_item(item, tree_node.right))

def evaluate_array(input_arr, tree_node, outfile):
    correct = 0
    incorrect = 0
    arr_no_names = input_arr[1:]
    with open(outfile, 'w') as file:
        for row in arr_no_names:
            prediction = evaluate_item(row, tree_node)
            #print(prediction, end = ' ')
            #print(row[-1])
            if (prediction == row[-1]):
                correct +=1
            else:
                incorrect +=1

            file.write(str(prediction) + '\n')
    file.close
    return (incorrect/(correct+incorrect))

def print_tree(node, buffer):
    #if(node.attr != None):
        print("["+str(node.num0)+" 0/"+str(node.num1)+ " 1]")
        if(node.left != None):
           print(buffer + node.attr + " = 0: ", end= "")
           print_tree(node.left, buffer+"| ")
        if(node.right != None):
           print(buffer + node.attr + " = 1: ", end= "")
           print_tree(node.right, buffer+"| ")

def write_metrics(train_error, test_error, outfile):
    with open(outfile, 'w') as file:
        file.write('error(train): '+ str(train_error) + '\nerror(test): ' + str(test_error))
    file.close
    return

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, 
                        help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    parser.add_argument("metrics_out", type=str, 
                        help='path of the output .txt file to which metrics such as train and test error should be written')
    args = parser.parse_args()


train_data_arr = np.genfromtxt(args.train_input, delimiter="\t", dtype=None, encoding=None)

test_data_arr = np.genfromtxt(args.test_input, delimiter="\t", dtype=None, encoding=None)



def execute_tree():
    new_tree= train_model(train_data_arr, 0)
    train_error = evaluate_array(train_data_arr, new_tree, args.train_out)
    test_error = evaluate_array(test_data_arr, new_tree, args.test_out)
    write_metrics(train_error, test_error, args.metrics_out)

    print_tree(new_tree, "| ")

execute_tree()
#python decision_tree.py small_train.tsv small_test.tsv 3 output/test.txt output/train.txt output/metrics.txt