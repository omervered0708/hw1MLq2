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
        self.sorted_index_lex = np.argsort(self.training_y)
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        # get vectorized version of predict single to use on the given array
        v_predict_single = np.vectorize(self.predict_single)
        # map the predict_single function onto X to get prediction vector
        predictions = np.array(v_predict_single(X))
        return predictions

    def predict_single(self,test_point):
        """
        predicts the label using the classifier for a single point
        :param test_point: the point to predict the label
        :return: predicted label
        """
        sorted_dist_index, distances = self.sort_index_by_dist(test_point)
        sorted_indices_both = np.lexsort((self.sorted_index_lex, sorted_dist_index))
        sorted_labels = self.training_y[sorted_indices_both]
        k_neighborhood_labels = sorted_labels[:self.k]

    def sort_index_by_dist(self, point):
        """
        :param point: a numpy array representing a point in dataset
        :return: an array of indices of training_x sorted by distances from point
        """
        distances = np.vectorize(lambda x: self.p_norm(point=(point-x), p=self.p))(self.training_x)
        sort_index = np.argsort(distances)
        return sort_index, distances

    @staticmethod
    def p_norm(point, p):
        """
        :param point: np.array to calculate p-norm of
        :param p: positive number
        :return: p-norm of point ||point||_p
        """
        v_abs_p_pow = np.vectorize(lambda x_i: abs(x_i)**p)
        return (np.sum(v_abs_p_pow(point)))**(1/p)


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
