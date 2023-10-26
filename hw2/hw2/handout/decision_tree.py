import argparse
import sys
import numpy as np
import math

class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None


def Plog2P(probability):
    if (probability == 0):
        return 0.0
    return(probability * math.log2(probability))

#this is H(Y) where P1 and P2 are the counts of Yes and No
def label_entropy(count1, count2):
    P1 = count1/(count1+count2)
    P2 = count2/(count1+count2)
    return -(Plog2P(P1)+Plog2P(P2))

def conditional_entropy(feature_name, names_list, input_arr):

    col_index = names_list.index(feature_name)
    #sorting items into arrays where the specified feature is 1 or 0. these will be passed to child nodes later
    feature_0_arr = input_arr[input_arr[:,col_index] == 0]
    feature_1_arr = input_arr[input_arr[:,col_index] == 1]
 
    #using the size of each arr to determine how many each of given feature are 0/1
    num_0s = feature_0_arr.shape[0]
    num_1s = feature_1_arr.shape[0]


    #extracting label 0/1 output given feature is 0
    Y_counts_feature_0 = np.unique(feature_0_arr[:,-1:], return_counts=True)[1]
    #for given feature X, P(X=0)
    P_feature_0 = (num_0s/(num_0s+num_1s))
    #for label Y and feature X = 0/1, this is H(Y|X=0)
    #if-else to account for pure nodes, np.unique will be different
    if num_0s==0:
        entropy_given_feature_0 = 0
    elif type(Y_counts_feature_0[0]) is np.int64:
        entropy_given_feature_0 = 0
    else:
        entropy_given_feature_0 = label_entropy(Y_counts_feature_0[0],Y_counts_feature_0[1])


    #H(Y|X=1)
    Y_counts_feature_1 = np.unique(feature_1_arr[:,-1:], return_counts=True)[1]
    P_feature_1 = (num_1s/(num_0s+num_1s))
    if num_1s==0:
        entropy_given_feature_1 = 0
    elif type(Y_counts_feature_1[0]) is np.int64:
        entropy_given_feature_1 = 0
    else:
        entropy_given_feature_1 = label_entropy(Y_counts_feature_1[0],Y_counts_feature_1[1])

    #this is P(X=0)H(Y|X=0)+P(X=1)H(Y|X=1)
    cond_entropy = (P_feature_0*entropy_given_feature_0 + P_feature_1*entropy_given_feature_1)
    return cond_entropy
    #for specific feature X
    #calculate label_entropy of X = 0 and X = 1
    #to do this:
    #obtain counts of Y = 0 and Y = 1 FOR X = 0
    #label_entropy(countY0, countY1) for x = 0
    #sum with label_entropy(CountY0, countY1) for x = 1

def mutual_information(feature_name, names_list, input_arr):
    #setting up counts to calculate H(Y)
    Y_counts_total = np.unique(input_arr[:,-1:], return_counts=True)[1]
    #return label_entropy() - conditional_entropy(feature_name, names_list, input_arr)
    #this is H(Y)
    if (len(Y_counts_total) == 1):
        return (0-conditional_entropy(feature_name, names_list, input_arr))
    
    mut_info = (label_entropy(Y_counts_total[0], Y_counts_total[1])-conditional_entropy(feature_name, names_list, input_arr))
    return mut_info

def find_splitting_feature(names_list, input_arr):
    info_per_feature = []
    for name in names_list[:-1]:
        info_per_feature.append(mutual_information(name, names_list, input_arr))
    max_info = max(info_per_feature)
    split_feature = names_list[info_per_feature.index(max_info)]        
    #print("feature to split:"+split_feature)
    return (split_feature, max_info)

def pare_dataset(removed_feature, names_list, input_arr):

    col_index = names_list.index(removed_feature)
    new_names = list(names_list)
    new_names.pop(col_index)

    return(new_names, np.delete(input_arr, col_index, 1))

#start w/ tree = train_model(names_list, input_arr, depth``)
def train_model(names_list, input_arr, depth):
    if(np.size(input_arr)==0):
        return
    if (depth >= args.max_depth) or (len(names_list) ==1 ): #if last feature or maxdepth
        leaf_node = Node()
        leaf_node.vote = majority_vote(input_arr)
        #print("leaf generated at depth "+str(depth)+ " with vote "+str(leaf_node.vote))
        return leaf_node
    
    #print("depth: "+ str(depth))

    #find the feature to split upon
    splitting_feature, max_info = find_splitting_feature(names_list, input_arr)
    if max_info == 0:
        leaf_node = Node()
        leaf_node.vote = majority_vote(input_arr)
        #print("leaf generated at depth "+str(depth)+ " with vote "+str(leaf_node.vote) +" due to 0 MI")
        return leaf_node
    col_index = names_list.index(splitting_feature)
    #split the data by the feature
    feature_0_arr = input_arr[input_arr[:,col_index] == 0]
    feature_1_arr = input_arr[input_arr[:,col_index] == 1]
    
    #pare each dataset
    f0_pared_names, f0_pared_arr = pare_dataset(splitting_feature, names_list, feature_0_arr)
    f1_pared_names, f1_pared_arr = pare_dataset(splitting_feature, names_list, feature_1_arr)
    
    #generate node
    split_node = Node()
    split_node.attr = splitting_feature

    #print("generating left subtree at depth " + str(depth+1))
    split_node.left = train_model(f0_pared_names, f0_pared_arr, depth+1)
    #print("generating right subtree at depth " + str(depth+1))
    #print("current split: " + splitting_feature)
    split_node.right = train_model(f1_pared_names, f1_pared_arr, depth+1)
    #return node
    return split_node

def majority_vote(input_arr):

    sliced_arr = input_arr[:, -1:]
    last_col_summary = np.unique(sliced_arr, return_counts=True)
    if(len(last_col_summary[0])==1):
        return last_col_summary[0][0]
    majority_vote = last_col_summary[0][last_col_summary[1].argmax()]
    if last_col_summary[1][0]==last_col_summary[1][1]:
        if last_col_summary[0][0]>last_col_summary[0][1]:
            majority_vote = last_col_summary[0][0]
        else:
            majority_vote = last_col_summary[0][1]

    #error_rate = 1- (last_col_summary[1].max()/(np.sum(last_col_summary[1])))
    return(majority_vote)

def evaluate_item(names_list, item, tree_node):
    if(tree_node.vote != None):
        #print("result found: "+str(tree_node.vote))
        return tree_node.vote
    #print("evaluating "+ tree_node.attr)
    col_index = names_list.index(tree_node.attr)
    if (item[col_index] == 0) and (tree_node.left != None):
        #print("searching left...")
        return(evaluate_item(names_list, item, tree_node.left))
    if (item[col_index] ==1) and (tree_node.right != None):
        #print("searching right...")
        return(evaluate_item(names_list, item, tree_node.right))

def evaluate_array(names_list, input_arr, tree_node, outfile):
    correct = 0
    incorrect = 0
    with open(outfile, 'w') as file:
        for row in input_arr:
            prediction = evaluate_item(names_list, row, tree_node)
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
    if(node.attr != None):
        print(buffer+node.attr)
    elif(node.vote != None):
        print(buffer+str(node.vote))

    if(node.left != None):
        print_tree(node.left, (buffer+"|"))
    
    if(node.right != None):
        print_tree(node.right, (buffer+"|"))
    return

def write_metrics(train_error, test_error, outfile):
    with open(outfile, 'w') as file:
        file.write('error(train): '+ str(train_error) + '\nerror(test): ' + str(test_error))
    file.close
    return
    
'''
    Mutual information for a feature pseudocode:
    1. H(Y)- count the 0s and 1s in label, calculate H(Y) using entropy formula
    2. H(Y|X1 = x1) -this is the info of a specific feature having a specific value. For example, if weather is X1 this is for 
                weather = sun specifically. Same equation as 1, but only use values for the specific value of the feature
                ex: -1[P(no|sun)log2P(no|sun)+ P(yes|sun)log2P(yes|sun)]
    3. H(Y|X1) -note, steps 2 and 3 aren't necessarily for X1, X1 is an example but could be any feature.
            Simply add up different values for each feature. For H(Y|weather) you would have:
            [P(sun)H(Y|sun) + P(cloud)H(Y|cloud) + P(rain)H(Y|rain)] etc.
'''

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


train_data_arr = np.genfromtxt(args.train_input, delimiter="\t", skip_header=1)
test_data_arr = np.genfromtxt(args.test_input, delimiter="\t", skip_header=1)


with open(args.train_input) as f:
    first_line = f.readline()
    names = first_line.split()

print(mutual_information("thalassemia", names, train_data_arr))


decision_tree = train_model(names, train_data_arr, 0)
train_error = evaluate_array(names, train_data_arr, decision_tree, args.train_out)
test_error = evaluate_array(names, test_data_arr, decision_tree, args.test_out)
write_metrics(train_error, test_error, args.metrics_out)
