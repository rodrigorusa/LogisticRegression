import argparse
import numpy as np
import math
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Logistic Regression.')
parser.add_argument('-training', dest='training_path')
parser.add_argument('-test', dest='test_path')


def logistic_regression_one_vs_all():
    print("Starting Logistic Regression One-vs-All...")


def multinomial_logistic_regression():
    print("Starting Multinomial Logistic Regression...")


def scikit_logistic_regression():
    print("Starting Scikit Logistic Regression...")


def init_dataset():
    print("Initializing dataset...")


def main():
    args = parser.parse_args()

    init_dataset(args)

    print('Choose your method:')
    print('1 - Logistic Regression One-vs-All')
    print('2 - Multinomial Logistic Regression')
    print('3 - Scikit Logistic Regression')
    print('Anyone - Exit')

    option = int(input('Option: '))

    if option == 1:
        logistic_regression_one_vs_all()
    elif option == 2:
        multinomial_logistic_regression()
    elif option == 3:
        scikit_logistic_regression()


if __name__ == '__main__':
    main()
