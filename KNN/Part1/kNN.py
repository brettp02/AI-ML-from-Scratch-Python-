import numpy as np
from random import seed
from random import randrange
import csv
from math import sqrt
from statistics import mode
import sys



# Load a CSV file using numpy
def load_csv(filename):
    dataset = np.genfromtxt(filename, delimiter=",", skip_header=1)
    X = dataset[:, :-1] # all rows, all columns except the last one
    y = dataset[:, -1]  # all rows, only the last column (class)
    return X, y


# calculate euclidean distance between two vectors
def euclidean_distance(vector_a, vector_b):
    distance = 0.0
    for i in range(len(vector_b)-1):
        distance += (vector_a[i] - vector_b[i])**2
    return sqrt(distance)


# Locate the most similar neighbors
def get_kNN(X_train, y_train, test_instance, k):
    distances = list()
    for i in range(len(X_train)):
        dist = euclidean_distance(test_instance, X_train[i]) # calculate distance
        distances.append((X_train[i], y_train[i], dist)) # store the distance
    distances.sort(key=lambda tup: tup[2]) # sort by distance (ascending)
    neighbours = list()
    for i in range(k):
        neighbours.append((distances[i][0], distances[i][1])) # store the k nearest neighbours
    return neighbours


# Make a class prediction with kNN
def predict_class(X_train, y_train, test_row, k):
    neighbours = get_kNN(X_train, y_train, test_row, k) # get the k nearest neighbours
    output_values = [row[1] for row in neighbours] # store the class labels of the k nearest neighbours
    prediction = mode(output_values) # get the most common class label from kNN
    return prediction


# Min-Max normalization
def min_max_normalize(dataset):
    # Separate the features and class labels
    features = dataset[:, :-1] # all rows, all columns except the last one
    class_labels = dataset[:, -1].reshape(-1, 1) # all rows, only the last column (class)

    # Normalize the features
    min_values = features.min(axis=0) # minimum value for each column
    max_values = features.max(axis=0) # maximum value for each column
    ranges = max_values - min_values
    normalized_features = (features - min_values) / ranges # normalize the features

    # Combine the normalized features and class labels
    normalized_data = np.hstack((normalized_features, class_labels))

    return normalized_data


# Main function 
def main(train_file, test_file, output_file, k):
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 5:
        print("Usage: python script.py <train_file> <test_file> <output_file> <k>")
        return

    # Assign command-line arguments to variables
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]
    k = int(sys.argv[4])  # Convert k to an integer



    # Load and normalize training data
    X_train, y_train = load_csv(train_file)
    X_train = min_max_normalize(X_train)

    # Load and normalize testing data
    X_test, y_test = load_csv(test_file)
    X_test = min_max_normalize(X_test)

    # Initialize a list to store the predictions
    predictions = []

    # Open output file
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Original Class', 'Predicted Class', 'Distances'])

        # Make predictions for each test instance and write to file
        for i in range(len(X_test)): # for each test instance
            prediction = predict_class(X_train, y_train, X_test[i], k) # make a prediction
            predictions.append(prediction)  # Store the prediction
            neighbours = get_kNN(X_train, y_train, X_test[i], k) # get the k nearest neighbours
            distances = [euclidean_distance(X_test[i], neighbour[0]) for neighbour in neighbours] # store the distances
            writer.writerow([y_test[i], prediction] + distances) # write to file

    # Calculate accuracy
    correct_predictions = 0
    for i in range(len(predictions)):
        if predictions[i] == y_test[i]:
            correct_predictions += 1
    accuracy = correct_predictions / len(y_test)
    print(f"Predictions written to {output_file}")
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))







