import numpy as np
import pandas as pd
import csv
import sys

'''
Naive Bayes classifier

Based off of Bayes Theorem:
- P(A|B) = P(B|A) * P(A) / P(B)

It describes how to update our beliefs about the probability of an event based on new evidence which
we represent as P(A|B)

e.g. Probability that it will rain given that it is cloudy
'''

def load_training_data(path):
    df = pd.read_csv(path, delimiter=',')
    #split into X and y and convert to numpy array
    X = df.iloc[:, 2:].values # don't include class and number
    y = df['class'].values

    return X,y,df #Using df for calculations


class NaiveBayes:

    def __init__(self,X,y,df):
        self.X,self.y,self.df = X,y,df
        self.class_labels = np.array(['no-recurrence-events', 'recurrence-events'])
        self.features = self.df.columns[2:].values
        self.possible_values = { #hard-coded as couldn't get the non-used values (e.g. age 90-99) added using other methods
            'age': ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'],
            'menopause': ['lt40', 'ge40', 'premeno'],
            'tumor-size': ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49',
                           '50-54', '55-59'],
            'inv-nodes': ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30-32',
                          '33-35', '36-39'],
            'node-caps': ['yes', 'no'],
            'deg-malig': [1, 2, 3],
            'breast': ['left', 'right'],
            'breast-quad': ['left_up', 'left_low', 'right_up', 'right_low', 'central'],
            'irradiat': ['yes', 'no']
        }



    # Train the NaiveBayes classifier on training data in constructor
    # Using pseudocode from lectures
    # @return - probability table (prob)
    def train(self):
        #Initialise the count numbers to 1
        count = {} #Dict to store the class_label/count
        for y in self.class_labels:
            count[y] = 1
            for Xi in self.features:
                for xi in self.possible_values[Xi]:
                    count[(Xi,xi,y)] = 1


        # Count the numbers of each class and feature value based on the training instances
        training_instances = self.df.iloc[:,1:].values # used to have both X and y without the null feature at idx[0]

        for i in range(len(training_instances)):
            X_val = training_instances[i,1:]
            y_val = training_instances[i,0]

            count[y_val] += 1

            for j,Xi in enumerate(self.features):
                xi = X_val[j]
                count[(Xi,xi,y_val)] += 1


        # Calculate the total/denominators
        class_total = 0
        total = {}

        for y in self.class_labels:
            class_total += count[y]
            for Xi in self.features:
                total[(Xi,y)] = 0
                for xi in self.possible_values[Xi]:
                    total[(Xi,y)] += count[(Xi,xi,y)]


        # Calculate the probabilities from the counting numbers.
        self.prob = {}
        for y in self.class_labels:
            self.prob[y] = count[y]/class_total
            for Xi in self.features:
                for xi in self.possible_values[Xi]:
                    self.prob[(Xi,xi,y)] = count[(Xi,xi,y)]/total[(Xi,y)]

        return self.prob


    # Calculate the score of each test instance using probability table from training function
    def score(self, y, prob, X_instance):
        score = prob[y]
        for i, Xi in enumerate(self.features):
            xi = X_instance[i]
            score *= prob[(Xi, X_instance[i], y)]
        #print(score)
        return score



    # Predict the class labels of test instances, using probability table from training
    def predict(self, X_test):
        y_pred = []
        for instance in X_test: #each row
            scores = {}
            for y in self.class_labels: #recurrence/no-recurrence
                scores[y] = self.score(y, self.prob, instance) #calculate scores for recurrence and no-recurrence
            y_pred.append(max(scores, key=scores.get)) #add the higher score value to prediction e.g. recurrence score > no-recurrence then predict that it is recurrence
        return y_pred



    # Calculate accuracy - how often the model predicts correctly
    def calc_accuracy(self,y_true,y_pred):
        correct = 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]: #if predicted class was correct
                correct += 1

        accuracy = correct / len(y_true)
        return accuracy



def main(args):
    if len(args) != 3:
        print('Incorrect arguments. Should be: python3 NaiveBayes.py train.cst test.csv')

    #load training_files
    training_file = args[1]
    test_file = args[2]

    #Split into X test, Xtrain, y_test, y_train aswell as the overall dataframes for use in training function
    X_train,y_train,df_train = load_training_data(training_file)
    X_test, y_test, df_test = load_training_data(test_file)


    # train model on training data and make predictions with the probability table calculated from training
    nb = NaiveBayes(X_train,y_train,df_train)
    prob = nb.train()
    y_hat = nb.predict(X_test)
    accuracy = nb.calc_accuracy(y_test,y_hat)


    # write output file with test output, also includes data for question 3
    with open('test_output.txt', 'w') as f:

        f.write('Original Class, Predicted Class, Scores No Recurrence, Scores Recurrence, Input Vector\n')

        for i in range(len(X_test)):
            instance = X_test[i]
            scores = {y: nb.score(y, prob, instance) for y in nb.class_labels}
            prediction = y_hat[i]
            f.write(f'{y_test[i]}, {prediction}, {scores["no-recurrence-events"]}, {scores["recurrence-events"]}, {instance}\n')


    # Output confitianal probabilities of each class, and also each feature in each class Q1/2
    with open('probabilities.txt', 'w') as f:
        for key, value in nb.prob.items():
            f.write(f'{key}: {value}\n')

    #print("y_predicted: ",y_hat )
    print("\nAccuracy =", accuracy, " \n")
    print('Testsample output + data for question 3 is in \'test_ouput.txt\' in current directory.')
    print('Conditional probabilities/data for Q1 + 2 printed to \'probabilities.txt\' in current directory.  \n')




if __name__ == '__main__':
    main(sys.argv)