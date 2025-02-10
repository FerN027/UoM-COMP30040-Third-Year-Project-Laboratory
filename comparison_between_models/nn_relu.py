# This script takes spec file, training data and testing data as input, and:
    # build a neural network with ReLu activiation;
    # translate it into SMT formulas by looping and adding equations for each neuron in each hidden layer;
    # run GearOpt_BO.

import sys
import json
import skopt
import random
import numpy as np
import pandas as pd

from z3 import *
from scipy.stats import qmc
from skopt import Optimizer
from skopt.space import Real

# ensure GearOpt_BO importable from both main directory and sub-directory
sys.path.append('./')
sys.path.append('../')
import GearOpt_BO

from tensorflow import keras
from keras import layers, models


# read spec (.json) file, return lower and upper bounds, and radius lists for input features
def read_spec(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)

    # access the 'variables' array
    features = data['variables']

    # extract lower bound, upper bound and radius for each feature dimension
    lower_bounds, upper_bounds, radius_list = [], [], []

    for x_i in features:
        lower_bounds.append(x_i['bounds'][0])
        upper_bounds.append(x_i['bounds'][1])
        radius_list.append(x_i['radius'])

    return lower_bounds, upper_bounds, radius_list

# use closure to create function f(X_i) that uses the trained model to make prediction on feature vector X_i without passing the model itself
def create_f(trained_model):
    def f(X):
        # assuming X = [x_1, x_2, ..., x_n], which is used this way in GearOpt_BO
        return trained_model.predict(np.array([X]), verbose=0)[0][0]
    
    return f

# build a nn-relu model
def build_nn_relu(X_train, y_train, dimensions, num_layers, num_neurons_for_layers, epochs=100, batch_size=32):
    model = models.Sequential()

    # specify number of input features
    model.add(layers.Input(shape=(dimensions,)))

    # add hidden layers
    # assume relu activiation and num_neurons_for_layers a list specifying num of neurons for each hidden layer
    # len(num_neurons_for_layers) == num_layers
    for i in range(num_layers):
        model.add(layers.Dense(num_neurons_for_layers[i], activation='relu'))

    # add output layer
    model.add(layers.Dense(1))

    # compile and fit model
    model.compile(
        optimizer='adam', 
        loss='mean_squared_error'
    )

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    return model

if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.stderr.write("Usage: python3 polynomial.py [specifications (.json)] [training data (.csv)] [testing data (.csv)]\n")
        sys.exit(1)

    else:
        
        # 1.
        # preparation
        spec_file = sys.argv[1]
        train_file = sys.argv[2]
        test_file = sys.argv[3]

        # extract info needed for GearOpt_BO algorithm
        lower_bounds, upper_bounds, radius_list = read_spec(spec_file)
        dimensions = len(radius_list)

        # read training and testing samples
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)

        X_train = df_train.iloc[:, :-1].values
        y_train = df_train.iloc[:, -1].values
        
        X_test = df_test.iloc[:, :-1].values
        y_test = df_test.iloc[:, -1].values

        # check number of features in spec.json == number of features in samples.csv
        if (dimensions != len(X_train[0])) or (dimensions != len(X_test[0])):
            sys.stderr.write("Inconsistent number of features between .json and .csv\n")
            sys.exit(1)

        # 2.
        # train a nn-relu model
        num_layers = 2
        neurons_for_each_layer = [8, 16]

        model = build_nn_relu(X_train, y_train, dimensions, num_layers, neurons_for_each_layer)

        # 3.
        # now create the model's symbolic representation in z3
        z3_solver = z3.Solver()

        # add domain constraints for each feature
        z3_features = [z3.Real('x_' + str(i + 1)) for i in range(dimensions)]

        for i in range(dimensions):
            z3_solver.add(z3_features[i] >= lower_bounds[i])
            z3_solver.add(z3_features[i] <= upper_bounds[i])

        # next, translate nn-relu into z3 by looping each layer and each neuron in it, generate an equation and add into z3
        # naming convention: z3.Real('hi_nj') for j-th neuron in i-th hidden layer
        num_layers = len(model.layers)

        for i in range(num_layers):
            # weights:
                # [
                #   [weights of 1st input for all output]
                #   [weights of 2nd input for all output]
                #   ... ... ... ... ... ... ... ... ...
                #   [weights of ith input for all output]
                # ]
            # biases: vector containing bias term for each output
            weights, biases = model.layers[i].get_weights()

            num_input = weights.shape[0]
            num_output = weights.shape[1]

            if i == 0:
                # first layer with:
                    # input: input features
                    # output: neurons in the first hidden layer
                hidden_layer_index = 1

                # calculate each neuron and add them into z3
                for jth_neuron in range(num_output):
                    all_terms = [weights[ith_feature][jth_neuron] * z3_features[ith_feature] for ith_feature in range(num_input)] + [biases[jth_neuron]]
                    
                    # apply relu as if-else in z3
                    z3_solver.add(z3.Real('h' + str(hidden_layer_index) + '_n' + str(jth_neuron + 1)) == z3.If(sum(all_terms) > 0, sum(all_terms), 0))

            elif i == num_layers - 1:
                # last layer with:
                    # input: neurons in last hidden layer
                    # output: y
                hidden_layer_index = i

                all_terms = [weights[jth_neuron][0] * z3.Real('h' + str(hidden_layer_index) + '_n' + str(jth_neuron + 1)) for jth_neuron in range(num_input)] + [biases[0]]

                # no relu on the final output y
                z3_solver.add(z3.Real('y') == sum(all_terms))

            else:
                # hidden layers with:
                    # input: neurons in previous hidden layer
                    # output: neurons in current hidden layer
                prev_layer_index = i
                curr_layer_index = i + 1

                for jth_curr in range(num_output):
                    all_terms = [weights[ith_prev][jth_curr] * z3.Real('h' + str(prev_layer_index) + '_n' + str(ith_prev + 1)) for ith_prev in range(num_input)] + [biases[jth_curr]]

                    # add equation for jth_curr output neuron into z3
                    z3_solver.add(z3.Real('h' + str(curr_layer_index) + '_n' + str(jth_curr + 1)) == z3.If(sum(all_terms) > 0, sum(all_terms), 0))

        # 4.
        # perform GearOpt_BO algorithm
        func = create_f(model)

        GearOpt_BO.gearopt_bo(func, z3_solver, z3_features, lower_bounds, upper_bounds, radius_list, 5, max_i_A_max=5)

