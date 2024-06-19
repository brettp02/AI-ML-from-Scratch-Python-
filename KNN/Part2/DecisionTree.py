import numpy as np
from collections import Counter
import pandas as pd
import sys

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*, value=None, ig = None, entropy = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.ig = ig
        self.entropy = entropy

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    # Fit the model
    def fit(self, X, Y): # X is the feature matrix, Y is the target vector
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self.build_tree(X, Y)


    # Build the tree
    def build_tree(self, X, y, depth=0): # X is the feature matrix, y is the target vector
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        #check stop conditions
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_indices = np.random.choice(n_features, self.n_features, replace=False) #randomly select features

        # find the best split
        best_thresh, best_feature, best_gain, entropy = self._best_split(X, y, feat_indices) #best split based on information gain
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        #create child nodes
        left_idx, right_idx = self._split(X[:, best_feature], best_thresh)
        left = self.build_tree(X[left_idx, :], y[left_idx], depth+1)
        right = self.build_tree(X[right_idx, :], y[right_idx], depth+1)
        return Node(best_feature, best_thresh, left, right, ig=best_gain, entropy=entropy)


    # Find the best split
    def _best_split(self, X, y, feat_indices): # X is the feature matrix, y is the target vector
        best_gain = -1
        split_idx, split_thresh = None, None
        best_entropy = None

        # iterate over all features
        for feat_idx in feat_indices:
            feature = X[:, feat_idx]
            thresholds = np.unique(feature)
            for threshold in thresholds:
                # calculate information gain
                gain = self._information_gain(y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
                    best_entropy = self._entropy(y)
        return split_thresh, split_idx, best_gain, best_entropy


    # Calculate information gain
    def _information_gain(self, y, feature, threshold): # y is the target vector, feature is the feature vector
        parent_entropy = self._entropy(y)
        left_idx, right_idx = self._split(feature, threshold)

        # if no split is possible
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        # calculate entropy of children
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r  # weighted average of children entropy
        ig = parent_entropy - child_entropy
        return ig


    # Calculate entropy
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0]) # entropy formula


    # Split the data
    def _split(self, feature, threshold): # feature is the feature vector, threshold is the threshold value
        left_idx = np.argwhere(feature <= threshold).flatten()
        right_idx = np.argwhere(feature > threshold).flatten()
        return left_idx, right_idx


    # Find the most common label
    def _most_common_label(self, y): # y is the target vector
        counter = Counter(y) # count the number of each class
        return dict(counter) # return the class with the highest count


    # Predict the target vector
    def predict(self, X): # X is the feature matrix
        return [self._traverse_tree(x, self.root) for x in X] # traverse the tree for each feature vector, and return the predicted class


    # Traverse the tree
    def _traverse_tree(self, x, node): # x is the feature vector, node is the current node
        if node.is_leaf():
            most_common_class = max(node.value.items(), key=lambda item: item[1])[0]
            return most_common_class
        if x[node.feature] <= node.threshold: # if the feature value is less than the threshold, go left
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


    # Print the tree
    def print_tree(self, node=None, depth=0,parent_feature = None, edge_value=None):
        if node is None:
            node = self.root

        indent = "  " * depth

        # Print the condition leading to this node if it's not the root and if edge_value is provided
        if edge_value is not None and parent_feature is not None:
            print(f"{indent}--att{parent_feature} == {edge_value}--")

        if node.is_leaf():
            all_classes = {class_label: node.value.get(class_label, 0) for class_label in
                           range(2)}
            counts = ', '.join([f"{cls}: {count}" for cls, count in all_classes.items()])
            print(f"{indent}leaf{{{counts}}}")
        else:

            if node.ig is not None and node.entropy is not None:
                print(f"{indent}att{node.feature} (IG:{node.ig:.4f}, Entropy: {node.entropy:.4f})")
            # Recursively print left and right child nodes, with incremented depth
            if node.left is not None or node.right is not None:
                self.print_tree(node.left, depth + 1, node.feature, 0)
                self.print_tree(node.right, depth + 1, node.feature, 1)



# Calculate the accuracy
def accuracy(y_true, y_pred): # y_true is the true target vector, y_pred is the predicted target vector
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# Load the data
def load_data(file):
    data = pd.read_csv(file)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y


def main():
    # Load the data
    train_file = sys.argv[1]
    output_file = sys.argv[2]
    X, Y = load_data(train_file)
    classifier = DecisionTree()
    classifier.fit(X, Y)
    y_pred = classifier.predict(X)
    acc = accuracy(Y, y_pred)
    print(f"Accuracy: {acc}")

    import io
    from contextlib import redirect_stdout

    tree_str = io.StringIO()
    with redirect_stdout(tree_str): # redirect the output to the string buffer
        classifier.print_tree()

    with open(output_file, 'w') as f:
        f.write(tree_str.getvalue())


if __name__ == "__main__":
    main()

#Note: code structure/class structure was based off and built up from the video https://www.youtube.com/watch?v=NxEHSAfFlK8&t=7s
    
