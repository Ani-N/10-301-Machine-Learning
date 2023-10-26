import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map

def get_frequencies(untrimmed_review):
    words_list = untrimmed_review.split()
    frequencies = {}
    for word in words_list:
        if word in glove_map:
            if word not in frequencies:
                frequencies[word] = 0
            frequencies[word] += 1
    return frequencies

def load_evaluate_reviews(file):
    out_array = np.empty(301)
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            row_frequencies = (get_frequencies(row[1]))
            label_feature_row = np.append(np.float64(row[0]), evaluate_feature(row_frequencies))
            out_array = np.vstack([out_array, label_feature_row])
    out_array = (np.around(np.delete(out_array, 0, axis=0), 6))
    return out_array

def evaluate_feature(word_frequencies):
    total_words = 0
    feature_vector = np.zeros(shape= 300)
    for word, frequency in word_frequencies.items():
        feature_vector = np.add(feature_vector, glove_map[word]*frequency) 
        total_words += frequency
    feature_vector = feature_vector*(1/total_words)
    return feature_vector

def write_formatted(formatted_array, file_out):
    print(formatted_array.astype(dtype= str))
    np.savetxt(fname=file_out, X=formatted_array, delimiter="\t", fmt='%.6f', newline="\n")
    return

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()

####################################################################################################
glove_map = load_feature_dictionary(args.feature_dictionary_in)

train_formatted = load_evaluate_reviews(args.train_input)
test_formatted = load_evaluate_reviews(args.test_input)
val_formatted = load_evaluate_reviews(args.validation_input)

write_formatted(train_formatted, args.train_out)
write_formatted(val_formatted, args.validation_out)
write_formatted(test_formatted, args.test_out)