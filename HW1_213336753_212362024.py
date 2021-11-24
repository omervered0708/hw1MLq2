import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class KnnClassifier:
    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.

        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """
        self.k = k
        self.p = p
        self.training_x = None
        self.training_y = None
        self.sorted_index_lex = None
        # TODO - Place your student IDs here. Single submitters please use a tuple like so: self.ids = (123456789,)
        self.ids = (213336753, 212362024)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.
        """
        self.training_x = X
        self.training_y = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        # map the predict_single function onto X to get prediction vector
        predictions = np.apply_along_axis(self.predict_single, 1, X)
        return predictions

    def predict_single(self, test_point):
        """
        predicts the label using the classifier for a single point
        :param test_point: the point to predict the label
        :return: predicted label
        """
        # get array of distances between the given point and all other points in the training set
        distances = self.get_distances(test_point)
        # get sorted indices array - primary sort by distance, secondary sort is lexicographically by label
        sorted_indices_both = np.lexsort((self.training_y, distances))
        # get the labels according the sorted indices array
        sorted_labels = self.training_y[sorted_indices_both]
        # take only first k labels
        k_neighborhood_labels = sorted_labels[:self.k]
        # get prediction
        return self.get_winning_label(k_neighborhood_labels)

    def get_distances(self, point):
        """
        :param point: a numpy array representing a point in dataset
        :return: an array of indices of training_x sorted by distances from point
        """
        # get array containing p-distances between the given point and the points in the training set
        distances = np.apply_along_axis(np.linalg.norm, 1, self.training_x - point, self.p)
        return distances

    @staticmethod
    def get_winning_label(k_neighbors_labels):
        """
        :param k_neighbors_labels: array with indices corresponding to points from the training set, sorted by distance
        to a point and secondarily lexicographically sorted by label.
        :return: predicted label according to the conditions specified (including tiebreaker conditions)
        """
        # find most frequent classes
        classes, counts = np.unique(k_neighbors_labels, return_counts=True)
        max_count = np.max(counts)
        frequent_classes = classes[np.where(counts == max_count)]
        # because the 'k_neighbors_labels' array is sorted by both ascending distance and ascending label, the first
        # occurrence of an element labeled as one of the most frequent classes in the k-neighborhood satisfies all
        # conditions in an event of a tiebreaker.
        # np.argmax will return the index of the first element in 'k_neighbors_labels' which belongs to one of the
        # frequent classes.
        return k_neighbors_labels[np.argmax(np.isin(k_neighbors_labels, frequent_classes))]


def main():

    print("*" * 20)
    print("Started HW1_ID1_ID2_old.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    parser.add_argument('k', type=int, help='k parameter')
    parser.add_argument('p', type=float, help='p parameter')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")

    print("Initiating KnnClassifier")
    model = KnnClassifier(k=args.k, p=args.p)
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)


if __name__ == "__main__":
    main()
