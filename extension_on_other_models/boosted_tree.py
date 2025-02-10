# This script extends the GearOpt_BO algorithm to boosted trees (sklearn.GradientBoostingRegressor) by translating it into SMT formulas in z3.

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
from sklearn.ensemble import GradientBoostingRegressor


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

# train a boosted tree model
def build_boosted_tree(X, y, n_estimators, learning_rate, max_depth):
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=0)

    model.fit(X, y)

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

# define a function that translates a decision tree into a list of z3.Implies(A, B), in which A is a path and B is the leaf value
def as_flatten_implications(all_paths, current_path, z3_features, node_index, features, thresholds, values, left_children, right_children):
    
    """
    all_paths: a global list of all paths, each path represented as [z3.Implies((x_1 <= ... AND x_2 > ... AND ...), y == val)]. Must be defined before the function calls.
    
    current_path: list of decision nodes met so far. When a leaf is reached, current_path is transfered into all_paths
    """

    # if a leaf is reached, current_path is complete, turn it into a z3.Implies() and add to all_paths
    if left_children[node_index] == right_children[node_index]:
        leaf_value = values[node_index].item()

        all_paths.append(z3.Implies(z3.And(current_path), z3.Real('y') == leaf_value))

    # else, split to its left and right child node, update their current_path respectively, and call the function recursively
    else:
        # get the feature id at this decision node and the threshold value
        decision_feature_index = features[node_index]
        decision_threshold = thresholds[node_index]

        # update current_path and node_index for both left and right child nodes
        left_path = current_path + [z3_features[decision_feature_index] <= decision_threshold]
        right_path = current_path + [z3_features[decision_feature_index] > decision_threshold]

        as_flatten_implications(all_paths, left_path, z3_features, left_children[node_index], features, thresholds, values, left_children, right_children)
        as_flatten_implications(all_paths, right_path, z3_features, right_children[node_index], features, thresholds, values, left_children, right_children)

# create an independent copy of the input solver
def copy_solver(s):
    copy = z3.Solver()

    for assertion in s.assertions():
        copy.add(assertion)

    return copy

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: python3 boosted_tree.py [specifications (.json)] [data (.csv)]\n")
        sys.exit(1)

    else:
        
        # 1.
        # preparation
        spec_file = sys.argv[1]
        data_file = sys.argv[2]

        # extract info needed for GearOpt_BO algorithm
        lower_bounds, upper_bounds, radius_list = read_spec(spec_file)
        dimensions = len(radius_list)

        # read data samples
        df = pd.read_csv(data_file)

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # check number of features in spec.json == number of features in samples.csv
        if (dimensions != len(X[0])):
            sys.stderr.write("Inconsistent number of features between .json and .csv\n")
            sys.exit(1)

        # 2.
        # train a boosted tree using GradientBoostingRegressor in sklearn
        model = build_boosted_tree(X, y, n_estimators=15, learning_rate=0.1, max_depth=3)

        # 3.
        # now create the model's symbolic representation in z3
        z3_solver = z3.Solver()

        # add domain constraints for each feature
        z3_features = [z3.Real('x_' + str(i + 1)) for i in range(dimensions)]

        for i in range(dimensions):
            z3_solver.add(z3_features[i] >= lower_bounds[i])
            z3_solver.add(z3_features[i] <= upper_bounds[i])

        # next, encode trained boosted tree into z3
        # get total number of short trees
        n_trees = len(model.estimators_)

        # loop through all sub-trees, encode them into z3 first
        for ith_tree in range(n_trees):
            # extract properties needed for SMT translation
            features = model.estimators_[ith_tree][0].tree_.feature
            thresholds = model.estimators_[ith_tree][0].tree_.threshold
            values = model.estimators_[ith_tree][0].tree_.value
            children_left = model.estimators_[ith_tree][0].tree_.children_left
            children_right = model.estimators_[ith_tree][0].tree_.children_right

            # since subtrees are not deep trees, use if-then-else representation here
            # naming convention: z3.Real('y1') refers to the prediction value made by the first sub-tree
            z3_solver.add(
                z3.Real('y' + str(ith_tree + 1)) == as_nested_if_then_else(
                    z3_features, 0, features, thresholds, values, 
                    children_left, children_right
                )
            )

        # now all sub-trees are encoded into z3, define the final output as:
            # y = initial_model + learning_rate * (y1 + y2 + ... + yn)
        initial_model = model.init_.predict(X.mean(axis=0).reshape(1, -1))[0]
        learning_rate = model.learning_rate

        z3_solver.add(
            z3.Real('y') == initial_model + 
            learning_rate * sum([z3.Real('y' + str(ith_tree + 1)) for ith_tree in range(n_trees)])
        )

        # check equivalence by comparing prediction made by model.predict() and z3.model()
        # for i in range(100):
        #     # generate random input within bounds
        #     l = [random.uniform(lower_bounds[i], upper_bounds[i]) for i in range(dimensions)]

        #     s = copy_solver(z3_solver)

        #     for j in range(dimensions):
        #         s.add(z3_features[j] == l[j])

        #     s.check()
        #     sol = float(s.model()[z3.Real('y')].as_fraction())

        #     print('model:', model.predict(np.array([l]))[0])
        #     print('z3:   ', sol)
        #     print()

        # 4.
        # now we have the SMT formulas of boosted tree, we can perform GearOpt_BO algorithm
        func = create_f(model)

        GearOpt_BO.gearopt_bo(func, z3_solver, z3_features, lower_bounds, upper_bounds, radius_list, 10, max_i_A_max=5)

