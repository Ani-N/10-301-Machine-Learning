import numpy as np
import math
import argparse

import matplotlib.pyplot as plt

def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)

def neg_log_likelihood(theta, X, y):
    sum = 0
    for i in range(y.shape[0]):
        sum += (y[i]* math.log(sigmoid(np.dot(theta, X[i]))))
    return (-1*sum/y.shape[0])


def extract_features(infile):
    in_array = np.genfromtxt(fname=infile, dtype= np.float64, delimiter='\t')
    #exctracting labels from formatted data
    y_labels = in_array[:,0]
    #folding bias term in vectors
    X_array = np.hstack((np.ones((in_array.shape[0], 1), np.float64), in_array[:,1:]))
    return y_labels, X_array

#defining dJ(i)(theta)/dTheta
def dJi_dTh(theta, X, y, i):
    theta_i = np.copy(theta)
    thetaT_X = np.matmul(theta_i, X[i])
    for j in range (theta_i.size):
        theta_i[j] = (sigmoid(thetaT_X)-y[i])*X[i][j]
    return theta_i

def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float) -> np.ndarray:
    NLL_tracker = np.ndarray(num_epoch)
# TODO: Implement `train` using vectorization
    for epoch in range(num_epoch):
        print(epoch)
        for i in range(y.size):
            theta = np.subtract(theta, (learning_rate*(dJi_dTh(theta, X, y, i))))
        NLL_tracker[epoch] = neg_log_likelihood(theta, X, y)
    return(theta, NLL_tracker)



def predict(
    theta : np.ndarray,
    X : np.ndarray
) -> np.ndarray:
    # TODO: Implement `predict` using vectorization
    prediction_labels = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        if sigmoid(np.dot(theta, X[i])) >= 0.5:
            prediction_labels[i] = 1
    return prediction_labels


def write_predictions(predictions, outfile):
    with open(outfile, 'w') as file:
        for i in range(predictions.shape[0]):
            file.write(str(predictions[i]) + '\n')
    file.close
    return

def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    # TODO: Implement `compute_error` using vectorization
    num_errors = 0.0
    for i in range(y.shape[0]):
        if y[i] != y_pred[i]:
            num_errors += 1.0
    return (num_errors/y.shape[0])

def write_error(train_error, test_error, outfile):
    with open(outfile, 'w') as file:
        file.write('error(train): '+ str(train_error) + '\nerror(test): ' + str(test_error))
    file.close
    return

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

train_labels, train_features = extract_features(args.train_input)
test_labels, test_features = extract_features(args.test_input)
val_labels, val_features = extract_features(args.validation_input)

theta = np.zeros(train_features.shape[1])
theta, y_axis_train = train(theta, train_features, train_labels, args.num_epoch, args.learning_rate)

theta = np.zeros(train_features.shape[1])
theta, y_axis_val = train(theta, val_features, val_labels, args.num_epoch, args.learning_rate)

#theta = np.zeros(train_features.shape[1])
#theta, y_axis_0_001 = train(theta, train_features, train_labels, args.num_epoch, 0.001)

#theta_val, val_y_axis = train(theta, val_features, val_labels, args.num_epoch, args.learning_rate)

train_predictions = predict(theta, train_features)
test_predictions = predict(theta, test_features)

x_axis = np.array(range(args.num_epoch))
fig, ax = plt.subplots()
ax.plot(x_axis, y_axis_train)
ax.plot(x_axis, y_axis_val)
#ax.plot(x_axis, y_axis_0_001)

train_error = compute_error(train_predictions, train_labels)
test_error = compute_error(test_predictions, test_labels)
write_error(train_error, test_error, args.metrics_out)
write_predictions(train_predictions, args.train_out)
write_predictions(test_predictions, args.test_out)

plt.show()


#small dataset: python lr.py traout.tsv valout.tsv tesout.tsv trainpreds.txt testpreds.txt metout.txt 500 0.1
#large dataset: python lr.py formatted_train_