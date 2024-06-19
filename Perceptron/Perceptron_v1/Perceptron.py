import pandas as pd
import numpy as np
import sys


#load the data and return X and y
def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=' ')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


'''
Perceptron class

@param - file_path = name/path to the data
@param - learning_rate, learning rate for Perceptron, have chosen 0.01 after testing
'''
class Perceptron:
    def __init__(self, file_path, learning_rate=0.01):
        self.lr = learning_rate
        self.X, self.y = load_data(file_path)
        self.bias = 1
        self.X = np.append(self.X, np.ones((self.X.shape[0], 1)), axis=1) #dummy deature
        self.weights = np.random.rand(len(self.X[0])) # random weight between 0-1
        self.accuracy = 0
        self.n_iters = 0

    #

    # predict if instance is in the good or bad class, return the classifier, unit step func
    def predict(self, X):
        linear_product = np.dot(X, self.weights) + self.bias
        return np.where(linear_product > 0, 'g','b')

    # Fit the model, implement algorithm from class
    def fit(self):
        learning_rate = self.lr
        max_iterations = 100
        iterations_without_progress = 0

        # while less than 1000 iterations and there hasn't been over 100 iterations without change, and accuracy less than 0.95(rounded 2dp)
        while self.n_iters < 1000 and iterations_without_progress < max_iterations and self.accuracy < 0.95:
            last_accuracy = self.get_accuracy()
            for i in range(len(self.X)):
                if self.predict(self.X[i]) != self.y[i]: # If perceptron is not correct
                    if self.y[i] == 'g': #if +ve example and wrong
                        self.weights += self.X[i] * learning_rate #add feature vector to weight vector
                    else: #if -ve and wrong
                        self.weights -= self.X[i] * learning_rate #substract feature vector from weight vector

            self.n_iters += 1 # next iteration
            self.accuracy = self.get_accuracy()
            print(f"Iteration {self.n_iters}, Accuracy: {self.accuracy:.3f}")

            #check for the max_iterations variable
            if self.accuracy == last_accuracy:
                iterations_without_progress += 1
            else:
                iterations_without_progress = 0


    # Return accuracy of perceptron
    def get_accuracy(self):
        accuracy = sum(1 for i in range(len(self.X)) if self.predict(self.X[i]) == self.y[i]) / len(self.X)
        return round(accuracy, 2)  # rounds to 2 decimal places to hit 95%
        #return accuracy # if you want to see without rounding



def main():

    import time
    st = time.time()
    if len(sys.argv) != 2:
        print("enter: python3 Perceptron.py ionosphere.data")
        sys.exit(1)

    filename = sys.argv[1]
    p = Perceptron(filename)
    p.fit()
    print('\n')
    print("Number of Training Iterations to Convergence: ", p.n_iters)
    print("Number of Misclassified Instances: ",
          len(p.X) - sum(1 for i in range(len(p.X)) if p.predict(p.X[i]) == p.y[i]),)
    et = time.time()
    print("Bias: ", p.bias)
    print("Final Accuracy: ", p.accuracy)
    time = et-st
    ms = time * 1000
    print(f"Time for {p.n_iters} iterations: {round(ms,1)}ms")
    print("Weight set: ", p.weights)


if __name__ == "__main__":
    main()

