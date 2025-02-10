# Implementation of the algorithm that combines Bayesian optimisation and SMT solver to find the maximised region of a ML model.

# See the main function gearopt_bo():

# Input:
#     trained ML model f(X);
#     its symbolic representation (SMT formulas in z3);
#     lower and upper bound for each feature;
#     radius for each feature (to decide whether a point is inside wrt this feature's dimension);
#     epsilon, the accuracy level for the bound of threshold;

# Output:
#     maximised threshold T, in the range [l, l + epsilon);
#     centers of safe regions whose values >= T everywhere;

import sys
import skopt
import random
import numpy as np
import pandas as pd


from z3 import *
from scipy.stats import qmc
from skopt import Optimizer
from skopt.space import Real


# Resolve incompatibility between Skopt and Numpy
np.int = int


# Implement BO solver for maximising f(x)
class A_max:
    
    """
    Since skopt.Optimizer is minimising by default, this maximiser is implemented by minimising the negation of the original function.
    Thus, when updating the maximiser by point (X_i, y_i), use maximiser.tell(X_i, -y_i).

    Parameter:

    known_candidates_and_counter_examples_list has form:
        [(C_1, y_1), (C_2, y_2), ..., (C_i, y_i), (D_1, z_1), (D_2, z_2), ..., (D_j, z_j)],
        where C_i, D_j are lists: [x_1, x_2, ..., x_n], instead of np.array.
    """

    def __init__(self, lower_bounds, upper_bounds, known_candidates_and_counter_examples_list):

        # vector of lower and upper bounds for each dimension
        self.a = lower_bounds
        self.b = upper_bounds

        bound_space = [Real(low, high) for low, high in zip(self.a, self.b)]

        self.solver = Optimizer(
            dimensions=bound_space,
            base_estimator='GP',
            acq_func='gp_hedge'
        )

        # use those known points to initialise the maximiser
        for point in known_candidates_and_counter_examples_list:
            self.solver.tell(point[0], -point[1])

    def suggest(self):
        # list form: [x_1, x_2, ..., x_n]
        return self.solver.ask()

    def observe(self, X_i, y_i):
        # X_i must be in the form: [x_1, x_2, ... ,x_n], not an np array
        self.solver.tell(X_i, -y_i)

# Implement BO solver for minimising f(x)
class B_min:
    def __init__(self, func, lower_bounds, upper_bounds, num_initial=20):

        # original function(ML model) to be minimised
        self.f = func

        # vector of lower and upper bounds for each dimension
        self.a = lower_bounds
        self.b = upper_bounds

        # number of points sampled to initialise the optimiser
        self.m = num_initial

        # use LHS sample heuristic to ensures a more even sample distribution
        self.samples = qmc.LatinHypercube(d=len(self.a))
        self.samples = self.samples.random(n=self.m)
        self.samples = qmc.scale(self.samples, self.a, self.b)

        # use those samples to initialise the Bayesian optimiser
        bound_space = [Real(low, high) for low, high in zip(self.a, self.b)]

        self.solver = Optimizer(
            dimensions=bound_space,
            base_estimator='GP',
            acq_func='gp_hedge'
        )

        for i in range(self.m):
            X_i = list(self.samples[i])
            y_i = self.f(X_i)
            self.solver.tell(X_i, y_i)

    def suggest(self):
        # list form:
        return self.solver.ask()

    def observe(self, X_i, y_i):
        # X_i must be a list: [x_1, x_2, ... ,x_n], not an np.array
        self.solver.tell(X_i, y_i)

# create an independent copy of the input solver
def copy_solver(s):
    copy = z3.Solver()

    for assertion in s.assertions():
        copy.add(assertion)

    return copy

# decide whether a point is in the region centered at another point
def is_nearby(center_point, around_point, radius_list):
    
    """
    center_point, around_point:
        [x_1, x_2, ..., x_n]

    radius_list:
        [r_1, r_2, ..., r_n]
    """
    
    dimensions = len(center_point)

    # check each feature dimension:
    for i in range(dimensions):
        # if any dimension is outside the allowed range for that dimension, return False
        if abs(center_point[i] - around_point[i]) > radius_list[i]:
            return False

    return True

# remove the region around a counter example from z3 solver's searching space
def exclude_region(counter_example_Xi, z3_solver, z3_features, radius_list):

    """
    counter_example_Xi:
        [d_1, d_2, ..., d_n]

    z3_features:
        [z3.Real('x_1'), z3.Real('x_2'), ..., z3.Real('x_n')]

    radius_list:
        [r_1, r_2, ..., r_n]
    """

    dimensions = len(counter_example_Xi)

    # if a point is inside this region, then it must be the case that:
        # (d_1 - r_1 <= x_1 <= d_1 + r_1) AND (d_2 - r_2 <= x_2 <= d_2 + r_2) AND ... AND (d_n - r_n <= x_n <= d_n + r_n)
    # whose negation (for a point not in this region) is equivalent to:
        # (x_1 < d_1 - r_1) OR (x_1 > d_1 + r_1) OR (x_2 < d_2 - r_2) OR (x_2 > d_2 + r_2) OR ... OR (x_n < d_n - r_n) OR (x_n > d_n + r_n)
    ors = []

    for i in range(dimensions):
        ors.append(z3_features[i] < counter_example_Xi[i] - radius_list[i])
        ors.append(z3_features[i] > counter_example_Xi[i] + radius_list[i])

    z3_solver.add(z3.Or(ors))

# Implement the algorithm that finds counter-examples which are less than T and within the region around the center candidate point:
def find_counter_example(func, z3_solver, z3_features, center_point, lower_bounds, upper_bounds, radius_list, T, max_iteration=100):

    """
    Parameters:

    func is the trained ML model, e.g. poly, nn, decision tree
        input: [x_1, x_2, ... , x_n]
        output: y
    
    z3_solver is the symbolic representation of the model, containing variables in z3.Real form with names: x_1, x_2,..., x_n, y, assuming n dimensions. It only has constraints on initial domains for each feature x_1, x_2, ..., x_n ,and the function realtion between y and [x_1, x_2, ..., x_n].

    z3_features: [z3.Real('x_1'), z3.Real('x_2'), ..., z3.Real('x_n')]

    center_point: a point in the feature space [c_1, c_2, ..., c_n], the region around which is to be examined.

    lower_bounds, upper_bounds, radius_list: (all lists) lower- and upper-bound and safe radius allowed for each feature.

    T: current threshold, any point around the center point that is evaluated lower than T is considered to be a counter example.

    max_iteration: max number of iterations allowed for BO search. If still no counter example found, SMT solver will be used.
    """

    # 1.
    # First, calculate the bounds for the searching space
    n = len(lower_bounds)
    search_lower_bounds, search_upper_bounds = [], []

    # for each feature
    for i in range(n):
        # check lower bound
        min = center_point[i] - radius_list[i]
        if min >= lower_bounds[i]:
            search_lower_bounds.append(min)
        else:
            search_lower_bounds.append(lower_bounds[i])

        # check upper bound
        max = center_point[i] + radius_list[i]
        if max <= upper_bounds[i]:
            search_upper_bounds.append(max)
        else:
            search_upper_bounds.append(upper_bounds[i])

    # 2.
    # initialise a B_min using the searching bounds from step.1 and LHS sampling heuristic
    BO_minimiser = B_min(func, search_lower_bounds, search_upper_bounds)
    
    # check first if any counter points are already in the initial points
    for point in BO_minimiser.samples:
        # point = [x_1 x_2 ... x_n]
        X = list(point)
        y = func(X)

        if y < T:
            # counter example found: ([x_1, x_2, ... x_n], y)
            print('counter-example found in BO')
            return (X, y)

    # otherwise, perform Bayesian optimisation (minimising) up to max_iteration times
    for i in range(max_iteration):
        # X_i is a list
        X_i = BO_minimiser.suggest()
        y_i = func(X_i)

        # check if it is a counter example
        if y_i < T:
            print('counter-example found in BO')
            return (X_i, y_i)

        BO_minimiser.observe(X_i, y_i)

    # 3.
    # if still no counter example found, use the SMT solver to do the formal satisfiability check on:
    # (x' in the region around center point) AND (y' = f(x')) AND (y' < T)
    # Since (y' = f(x')) has already been encoded into z3_solver, just add 1st and 3rd constraints into z3_solver
    
    # ensure each feature is in the allowed range centered at the center_point
    for i in range(n):
        z3_solver.add(z3_features[i] >= center_point[i] - radius_list[i])
        z3_solver.add(z3_features[i] <= center_point[i] + radius_list[i])

    # add the 3rd constraint
    z3_solver.add(z3.Real('y') < T)

    # check satisfiability
    if z3_solver.check() == unsat:
        return "unsat"
    
    else:
        # return a counter example, found by SMT solver
        sol = z3_solver.model()

        X = []
        for feature in z3_features:
            X.append(float(sol[feature].as_fraction()))

        y = float(sol[z3.Real('y')].as_fraction())

        print('counter-example found in z3')
        return (X, y)

# Algorithm that tries to find a potential candidate >= T at i-th iteration:
def find_candidate(func, maximiser, z3_solver, z3_features, radius_list, counter_examples_list, T, max_iteration=100):

    """
    Parameters:

    func:
        input [x_1, x_2, ..., x_n]
        return y
    
    maximiser: an A_max object

    z3_solver: symbolic representation of the ML model, containing constraints on initial domains for each feature x_1...x_n, function relation between y and X, previous generated lemmas for excluding regions around known counter examples, and y >= T, for the current T.

    z3_features: [z3.Real('x_1'), z3.Real('x_2'), ..., z3.Real('x_3')]

    radius_list: longest distance allowed away from center, for each feature

    counter_examples_list: [(D_1, z_1), (D_2, z_2), ..., (D_j, z_j)], counter examples found so far
        where D_j = [x_1, x_2, ..., x_n]

    T: current threshold, candidate should >= T

    max_iteration: max number of times of using BO, before z3 is used
    """

    # 1.
    # perform BO up to max_iteration times
    for i in range(max_iteration):
        X_i = maximiser.suggest()

        # first check if any known counter points are already in the region around this candidate point
        nearby_counter_example_values = []
        
        for point in counter_examples_list:
            if is_nearby(X_i, point[0], radius_list) and point[1] < T:
                nearby_counter_example_values.append(point[1])

        # if so, then X_i is a bad candidate, penalise its value to the minimal value among all known counter examples around it
        if len(nearby_counter_example_values) != 0:
            y_i = min(nearby_counter_example_values)

        # else, check its real function value
        else:
            y_i = func(X_i)

            # if y_i >= T, also since there's no known counter example near it, return (X_i, y_i) as a potential candidate
            if y_i >= T:
                print("candidate found in BO")
                return (X_i, y_i)
            
            # else, although no known counter example is near it, the point itself is a counter example
            # add it to list N and remove its region from z3 solver
            counter_examples_list.append((X_i, y_i))
            exclude_region(X_i, z3_solver, z3_features, radius_list)

        # update the maximiser
        maximiser.observe(X_i, y_i)

    # 2.
    # if still no valid candidate found, use z3 to do satisfiability check on:
    if z3_solver.check() == unsat:
        return "unsat"
    
    else:
        sol = z3_solver.model()

        X = []
        for feature in z3_features:
            X.append(float(sol[feature].as_fraction()))

        y = float(sol[z3.Real('y')].as_fraction())

        print('candidate found in z3')

        return (X, y)

# Main function:
def gearopt_bo(func, z3_solver, z3_features, lower_bounds, upper_bounds, radius_list, epsilon, max_i_A_max=10, max_i_B_min=10):

    """
    Parameters:

    func:
        input [x_1, x_2, ..., x_n]
        return y

    z3_solver: symbolic representation of the ML model, containing only constraints on initial domain for each feature, and function relation between y and X

    z3_features: [z3.Real('x_1'), z3.Real('x_2'), ..., z3.Real('x_3')]

    lower_bounds, upper_bounds: bounds lists for each feature

    radius_list: longest distance allowed away from center, for each feature

    epsilon: accuracy level, the final threshold should be bounded by [l, l + epsilon)

    max_i_A_max, max_i_B_min: max iteration times for BO maximiser and BO minimiser, respectively
    """

    # 1.
    # initialise global variables:
    
    # verified candidates list: [(C_1, y_1), (C_2, y_2), ...], where i-th candidate has safe region on i-th threshold
    candidates_list = []
    thresholds_list = []

    counter_examples_list = []

    l = float('-inf')
    u = float('inf')

    # arbitrary candidate lower and upper bounds l_0 and u_0, such that l_0 < u_0
    l_0 = -1
    u_0 = 1

    # 2.
    # main procedure
    while True:

        # 2.1
        # decide next threshold T to be examined using binary search based on value of l and u
        if u == float('inf'):
            T = u_0
            u_0 = 2 * u_0 - l_0

        elif l == float('-inf'):
            T = l_0
            l_0 = 2 * l_0 - u_0

        else:
            T = (l + u) / 2

        # 2.2
        # solve: whether a safe region exists, s.t. it evaluates to >= T everywhere within the region
            
        # initialise a BO maximiser using candidates and counter-examples found before
        BO_maximiser = A_max(lower_bounds, upper_bounds, candidates_list + counter_examples_list)

        # create an independent copy of z3_solver, to be used for candidate searching on current bound T
        # this is because regions around any counter-examples found are removed during the search
        z3_for_candidate_searching = copy_solver(z3_solver)
        z3_for_candidate_searching.add(z3.Real('y') >= T)

        print('\n-------------------------------------------------')
        print('Current bounds: l=', l, ' T=', T, ' u=', u)
        print('-------------------------------------------------\n')

        # begin the search
        while True:
            
            # try to find a potential candidate (C_i, y_i) s.t. no known counters around it AND y_i >= T
            candidate = find_candidate(
                func, BO_maximiser, z3_for_candidate_searching, z3_features, 
                radius_list, counter_examples_list, T, 
                max_iteration=max_i_A_max
            )

            # if unsat returned, then current T is too large, break the loop and reduce T
            if candidate == "unsat":
                u = T
                break

            # otherwise, we have a potential candidate point, use it as center to do counter-example check
            # Notice: use independent copy of the input solver this time, i.e. no constraints on searching space except features' inherent domain
            counter = find_counter_example(
                func, copy_solver(z3_solver), z3_features, 
                candidate[0], lower_bounds, upper_bounds, 
                radius_list, T, max_iteration=max_i_B_min
            )

            # if unsat returned (no counter-examples can be found), then it's a real candidate that defines a safe region >= T
            # record it, break the loop and try bigger threshold next
            if counter == "unsat":
                l = T

                candidates_list.append(candidate)
                thresholds_list.append(T)

                break

            # otherwise, a counter-example is found
            # this means although no known counters are near the candidate when it is returned, there is one counter found later
            # record it, penalise the BO maximiser, remove the region around it in SMT solver, and enter next iteration
            counter_examples_list.append(counter)
            BO_maximiser.observe(candidate[0], counter[1])
            exclude_region(counter[0], z3_for_candidate_searching, z3_features, radius_list)

        # check if l and u are closed enough
        if l == float('-inf') or u == float('inf'):
            continue

        # else, epsilon accuracy is achieved: u - l < epsilon
        elif u - l < epsilon:
            break

    # 3.
    # return all the center points defining regions >= l everywhere
    n = len(candidates_list)

    results = [(candidates_list[i][0], l) for i in range(n) if thresholds_list[i] >= l]

    print('\nResults found: centers of safe regions whose values >= T everywhere, where T is in [', l, ',', str(l + epsilon), '):')
    print(results)

    return results


