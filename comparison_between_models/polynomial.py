# This script takes a spec.json file, a training data set (.csv) and a testing data set (.csv), and:
    # build a polynomial model, whose degree is chosen based on the smallest mean squared error on testing data;
    # translate this model into SMT formulas (z3);
    # and run GearOpt_BO on it.

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

from functools import reduce
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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
# the returned function f is then passed into GearOpt_BO
def create_f(poly_model, poly_features):
    def f(X):
        # assuming X = [x_1, x_2, ..., x_n], which is used this way in GearOpt_BO
        return poly_model.predict(poly_features.transform(np.array([X])))[0]
    
    return f

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
        # train a polynomial model on training data:
        # since the poly degree cannot be set automatically, try different degrees and pick the one with the smallest mean squared error on testing data
        max_degree = 10
        mses = []

        for d in range(1, max_degree + 1):

            # Generating polynomial features, otherwise the linear regression is just a weighted sum of features
            # e.g. if [x_1, x_2] and degree=2:
                # polynomial features = [1, x_1, x_2, x_1^2, x_1*x_2, x_2^2]
            polynomial_features = PolynomialFeatures(degree=d)
            
            # transform original X_train and X_test into polynomial feature space
            X_train_poly = polynomial_features.fit_transform(X_train)
            X_test_poly = polynomial_features.transform(X_test)

            # training the model
            model = LinearRegression()
            model.fit(X_train_poly, y_train)

            # make predictions
            y_pred = model.predict(X_test_poly)

            # calculate the mse error
            mse = mean_squared_error(y_test, y_pred)

            mses.append(mse)

        # decide which degree to use and train the final model
        best_degree = np.argmin(mses) + 1

        polynomial_features = PolynomialFeatures(degree=best_degree)
        X_train_poly = polynomial_features.fit_transform(X_train)
        X_test_poly = polynomial_features.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # 3.
        # now create the model's symbolic representation in z3
        z3_solver = z3.Solver()

        # add domain constraints for each feature
        z3_features = [z3.Real('x_' + str(i + 1)) for i in range(dimensions)]

        for i in range(dimensions):
            z3_solver.add(z3_features[i] >= lower_bounds[i])
            z3_solver.add(z3_features[i] <= upper_bounds[i])

        # next, encode model function y=f(X) into z3
        y = z3.Real('y')

        # list of all poly features, starting at 1, e.g. x0^2 x1 x2
        poly_features_names = polynomial_features.get_feature_names_out()

        # coefficient for each poly feature
        coefficients = model.coef_

        # value when all input features are zeros
        intercept = model.intercept_

        # the polynomial function represented by this model is:
            # y = intercept + coefficients[0] * poly_features_names[0] + ... + coefficients[n] * poly_features_names[n]
        # next, translate poly_features_names into z3 format
        poly_features_in_z3 = []

        for name in poly_features_names:
            
            # check if it's first term, '1'
            if name == '1':
                poly_features_in_z3.append(float(name))

            # else it's in format e.g. 'x0^2 x1^3 x2', break into list elements and refer to the corresponding z3 variable
            else:
                # split the whole string into variables (str) list
                l = name.split(" ")
                
                for i in range(len(l)):
                    var = l[i]
                    
                    # since either var = "x(_i-th_)" or var = "x(_i-th_)^(_power_)"
                    # first, check if "^" is in string var
                    if '^' in var:
                        idx = var.find('^')
                        num = int(var[1:idx])
                        power = float(var[idx+1:])

                        l[i] = z3_features[num] ** power

                    else:
                        num = int(var[1:])

                        l[i] = z3_features[num]
                
                # now each var(str) is replaced by z3 expression, multiply them together 
                product = reduce(lambda x, y: x * y, l)
                
                poly_features_in_z3.append(product)

        # now all poly features are in legal z3 format, multiply each feature by their coefficient
        poly_features_in_z3 = [coefficients[i] * poly_features_in_z3[i] for i in range(len(poly_features_in_z3))]
        
        # lastly, add intercept and add the function representing the model into z3
        z3_solver.add(y == intercept + sum(poly_features_in_z3))

        # 4.
        # now we have the SMT representation of the model, we can perform the GearOpt_BO algorithm
        func = create_f(model, polynomial_features)
        
        GearOpt_BO.gearopt_bo(func, z3_solver, z3_features, lower_bounds, upper_bounds, radius_list, 0.1, max_i_A_max=3)
