import pandas as pd
import numpy as np
import sys


#load the data and return X and y
def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=' ')
    df = df.sample(frac=1, random_state=42)  # shuffles data for splitting
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


'''
Perceptron class

@param - file_path = name/path to the data
@param - learning_rate, learning rate for Perceptron, have chosen 0.01 after testing
'''
class Perceptron:
    def __init__(self, file_path, learning_rate=0.01, train_split=0.8):
        self.lr = learning_rate
        self.X, self.y = load_data(file_path)
        self.train_split = train_split
        self.split_data()
        self.bias = 1
        #self.X = np.append(self.X, np.ones((self.X.shape[0], 1)), axis=1)  # removed for test/train split version
        self.weights = np.random.rand(len(self.X_train[0]))  # random weight between 0-1
        self.accuracy = 0
        self.n_iters = 0

    # split data into X_train, X_test, y_train, y_test
    def split_data(self):
        split_inx = int(len(self.X) * self.train_split)
        self.X_train, self.X_test = self.X[:split_inx], self.X[split_inx:]
        self.y_train, self.y_test = self.y[:split_inx], self.y[split_inx:]
        self.X_train = np.append(self.X_train, np.ones((self.X_train.shape[0], 1)), axis=1)
        self.X_test = np.append(self.X_test, np.ones((self.X_test.shape[0], 1)), axis=1)


    # predict if instance is in the good or bad class, return the classifier, unit step func
    def predict(self, X):
        linear_product = np.dot(X, self.weights) + self.bias
        return np.where(linear_product > 0, 'g', 'b')


    # Fit the model, implement algorithm from class, modified to use X_train/y_train
    def fit(self):
        learning_rate = self.lr
        max_iterations = 100
        iterations_without_progress = 0

		# while less than 1000 iterations and there hasn't been over 100 iterations without change, and accuracy less than 0.95(rounded 2dp)
        while self.n_iters < 1000 and iterations_without_progress < max_iterations and self.accuracy < 0.95:
            last_accuracy = self.get_accuracy()
            for i in range(len(self.X_train)):
                if self.predict(self.X_train[i]) != self.y_train[i]: # If perceptron is not correct
                    if self.y_train[i] == 'g': #if +ve example and wrong
                        self.weights += self.X_train[i] * learning_rate #add feature vector to weight vector
                    else: #if -ve and wrong
                        self.weights -= self.X_train[i] * learning_rate #substract feature vector from weight vector

            self.n_iters += 1  #next iteration
            self.accuracy = self.get_accuracy()
            print(f"Iteration {self.n_iters}, Training Accuracy: {self.accuracy:.3f}")

			#check for the max_iterations variable
            if self.accuracy == last_accuracy:
                iterations_without_progress += 1
            else:
                iterations_without_progress = 0

    # Return accuracy of perceptron
    def get_accuracy(self):
        accuracy = sum(1 for i in range(len(self.X_train)) if self.predict(self.X_train[i]) == self.y_train[i]) / len(self.X_train)
        return accuracy  # rounds to 2 decimal places to hit 95%
        #return accuracy # if you want to see without rounding

    # Return accuracy of perceptron
    def evaluate(self):
        accuracy = sum(1 for i in range(len(self.X_test)) if self.predict(self.X_test[i]) == self.y_test[i]) / len(self.X_test)
        return accuracy

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
    print("Number of Misclassified Instances in Training: ",
          len(p.X_train) - sum(1 for i in range(len(p.X_train)) if p.predict(p.X_train[i]) == p.y_train[i]))
    print("Training Accuracy: ", p.accuracy)
    test_accuracy = p.evaluate()
    print("Test Accuracy: ", test_accuracy)
    et = time.time()
    print("Bias: ", p.bias)
    time = et - st
    ms = time * 1000
    print(f"Time for {p.n_iters} iterations: {round(ms, 1)}ms")
    print("Weight set: ", p.weights)

if __name__ == "__main__":
    main()