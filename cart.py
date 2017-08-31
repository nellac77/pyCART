# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:33:21 2017

@author: nellac77
"""
from random import seed
from random import randrange
from csv import reader

'''
This section will be for loading csv files, converting string columns to float,
splitting datasets into k folds, determing an accuracy percentage, and alorithm
evaluation by cross-validation split.
'''
# load a csv file
def load_csv(filename):
    f = open(filename, 'rb')
    lines = reader(f)
    dataset = list(lines)
    return dataset

# convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
        
# split the dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
        return correct / float(len(actual)) * 100.0



'''
This section will be to test then determine the splits in a given 
dataset. Then the Gini index will be calculated.
'''

# Split dataset based on an attribute and its value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    # note: right contains all rows with value at index
    # greater than or equal to split value
    return left, right

# Calculate the Gini index (cost function used to evaluate splits)
def gini_index(groups, classes):
    # count all samples at a split point
    n_instances = float (sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # no divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

# test Gini values
# Worst case, 0.5
#print (gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
# Best case, 0.0
#print (gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))

# Choose best split point for the dataset via exhaustive, greedy algorithm
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split((index), row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'score':b_score, 'groups':b_groups}

'''
The following section will be building the tree.
'''

# create the tree's terminal node (point where we know to stop building) value
def to_terminal(group):
    # select a class value for a group of rows
    outcomes = [row[-1] for row in group]
    # return most common output valuein list of rows
    return max(set(outcomes), key=outcomes.count)

# create child splits for a node, or make a terminal node
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no-split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left+right)
        return
    # check for max deoth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process the left child
    if len(left) <= to_terminal(left):
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process the right child
    if len(right) <= to_terminal(right):
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# build decision tree by creating the root node and use split() to recursively build
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# print the decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))
        
# make prediction with the decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
         if isinstance(node['right'], dict):
             return predict(node['right'], row)
         else:
             return node['right']

# test datasplitting process with a contirved dataset
dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]]

# look at the decision tree
#tree = build_tree(dataset, 1, 1)
#print_tree(tree)

#predict with a decision stump
stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
for row in dataset:
    prediction = predict(stump, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))
