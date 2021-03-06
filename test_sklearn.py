import argparse
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


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
    model = KNeighborsClassifier(n_neighbors=args.k, p=args.p)
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    # print("Fitting...")
    # model.fit(X, y)
    # print("Done")
    # print("Predicting...")
    # y_pred = model.predict(X)
    # print("Done")
    # accuracy = np.sum(y_pred == y) / len(y)
    # print(f"Train accuracy: {accuracy * 100 :.2f}%")
    # print("*" * 20)

    data_x, data_y = load_digits(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.25, random_state=42)
    print("Fitting...")
    model.fit(train_x, train_y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(test_x)
    print(y_pred)
    print("Done")
    accuracy = np.sum(y_pred == test_y) / len(test_y)
    print(f"test accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)


if __name__ == "__main__":
    main()
