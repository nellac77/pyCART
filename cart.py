# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:33:21 2017

@author: nellac77
"""
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
    return {'index ':b_index, 'value ':b_value, 'score ':b_score, 'groups ':b_groups}
