# This script takes user-defined inputs specifying number of feature dimensions and total number of samples, and do:
    # select the function (here it's polynomial from 1d to 5d) based on feature dimension input;
    # randomly generate some samples, divide into training samples (80%) and testing samples (20%);
    # output two .csv files ,respectively.

import sys
import csv
import random
import numpy as np

# sampling range for each feature
LOWER_BOUND = -10
UPPER_BOUND = 10

# define polynomial functions from 1d to 5d
# assuming X = [x_1, ..., x_n]
def f_1(X):
    return -X[0]**4 + 10*X[0]**3 - 25*X[0]**2 - 5*X[0] + 10

def f_2(X):
    return -X[0]**4 - X[1]**4 + 10*X[0]**2 + 10*X[1]**2

def f_3(X):
    return -X[0]**4 + X[0]**3 + X[1]**4 - X[1]**2 + X[2]**3 - X[2]**1 - 0.5*X[0]*X[1] + 0.5*X[1]*X[2] - 0.5*X[0]*X[2] + 100

def f_4(X):
    return -0.1*X[0]**4 + 0.5*X[0]**3 - X[1]**2 + X[1]*X[3]**2 + X[2]**2 - X[2]*X[3]**3 + 0.1*X[3]**4 - X[0]*X[1] + X[1]*X[2] - X[2]*X[3] + X[0]*X[3] + 150

def f_5(X):
    return (-0.03*X[0]**4 + 0.7*X[0]**3 - X[0]*X[1]**2 + 0.6*X[1]*X[4]**2 - 0.2*X[2]**3 + X[2]*X[3]**2 
            - 0.1*X[3]**4 + 0.5*X[3]*X[4]**3 - X[4]**2 + 0.8*X[0]*X[2] - 0.6*X[1]*X[3] + X[2]*X[4] 
            - 0.3*X[3]*X[1] + 0.4*X[0]*X[4]**2 - 0.2*X[4]*X[1]**2 + 250)

def samples_generator(d, n):
    total_samples = [list(X) for X in np.random.uniform(LOWER_BOUND, UPPER_BOUND, (n, d))]
    
    for X in total_samples:
        X.append(globals()[f'f_{d}'](X))

    return total_samples

def save_to_csv(filename, samples, d):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        first_row = ['x_' + str(i+1) for i in range(d)]
        first_row.append('y')

        csvwriter.writerow(first_row)
        csvwriter.writerows(samples)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: python3 random_samples_generator.py [number of dimensions] [total number of samples]\n")
        sys.exit(1)

    else:
        # extract inputs
        d = int(sys.argv[1])
        n = int(sys.argv[2])

        # check input dimension is supported
        if d not in [1, 2, 3, 4, 5]:
            sys.stderr.write("Dimension should be from 1 to 5\n")
            sys.exit(1)

        # get total samples
        total_samples = samples_generator(d, n)

        # split it: 80% for training and the 20% for testing
        cutting_index = int(n * 0.8)

        training_samples = total_samples[:cutting_index]
        testing_samples = total_samples[cutting_index:]

        # output two .csv files
        train_filename = str(d) + 'd_training_samples.csv' 
        test_filename = str(d) + 'd_testing_samples.csv'       

        save_to_csv(train_filename, training_samples, d)
        save_to_csv(test_filename, testing_samples, d)
