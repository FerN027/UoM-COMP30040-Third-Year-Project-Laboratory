# This script trains a decision tree on input data, translates it into SMT formulas as nested z3.If(), and runs GearOpt_BO algorithm.

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

from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


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
        return trained_model.predict(np.array([X]))[0]
    
    return f

def build_decision_tree(X_train, y_train, depth=20):
    model = DecisionTreeRegressor(max_depth=depth, random_state=0)

    model.fit(X_train, y_train)

    return model

# define a recursive function that translates a decision tree into z3 as nested if-then-else
def as_nested_if_then_else(z3_features, node_index, features, thresholds, values, left_children, right_children):
    # if current node is a leaf, return the value
    if left_children[node_index] == right_children[node_index]:
        return values[node_index].item()
    
    # else, encode it into a z3.If()
    else:
        return z3.If(
            z3_features[features[node_index]] <= thresholds[node_index],
            as_nested_if_then_else(z3_features, left_children[node_index], features, thresholds, values, left_children, right_children),
            as_nested_if_then_else(z3_features, right_children[node_index], features, thresholds, values, left_children, right_children)
        )

# create an independent copy of the input solver
def copy_solver(s):
    copy = z3.Solver()

    for assertion in s.assertions():
        copy.add(assertion)

    return copy

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
        # train
        model = build_decision_tree(X_train, y_train)

        # 3.
        # now create the model's symbolic representation in z3
        z3_solver = z3.Solver()

        # add domain constraints for each feature
        z3_features = [z3.Real('x_' + str(i + 1)) for i in range(dimensions)]

        for i in range(dimensions):
            z3_solver.add(z3_features[i] >= lower_bounds[i])
            z3_solver.add(z3_features[i] <= upper_bounds[i])

        # next, encode tree model into z3
        features = model.tree_.feature
        thresholds = model.tree_.threshold
        values = model.tree_.value
        children_left = model.tree_.children_left
        children_right = model.tree_.children_right

        # let y equal this nested if-then-else structure
        z3_solver.add(z3.Real('y') == as_nested_if_then_else(z3_features, 0, features, thresholds, values, children_left, children_right))

        # 4.
        # perform GearOpt_BO algorithm
        func = create_f(model)

        GearOpt_BO.gearopt_bo(func, z3_solver, z3_features, lower_bounds, upper_bounds, radius_list, 10, max_i_A_max=5)

        # for k in range(500): 

        #     l = [random.uniform(lower_bounds[i], upper_bounds[i]) for i in range(dimensions)]

        #     print('model:', func(l))

        #     s = copy_solver(z3_solver)

        #     for i in range(dimensions):
        #         s.add(z3_features[i] == l[i])

        #     s.check()
        #     sol = float(s.model()[z3.Real('y')].as_fraction())
        #     print('z3:   ', sol)
        #     print()
