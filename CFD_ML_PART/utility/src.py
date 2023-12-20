import catboost
from catboost import CatBoostClassifier
import datetime
from datetime import date
import glob
from itertools import permutations
import math
import matplotlib.pyplot as plt #has problem of importing in abac_evn
import numpy as np
import os
import pandas as pd

import random
import re

import sys
from sklearn import datasets
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.feature_selection import mutual_info_classif as MIC # the most relevant correlation to target variable, useful for classification problem
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler # base on each column/feature, it has mean 0 and SD 1
from sklearn.preprocessing import MinMaxScaler   # minimum value is 0 and maximum value is 1
from sklearn.tree import DecisionTreeClassifier




import torch
import torch.nn as nn
import time

import warnings
warnings.filterwarnings("ignore")
 

# support vector machine

def svm_classifier(x_dataset,y_label):
    """
    can be used for regression and classification
    kernel='linear', C=1                    # C is used to have missclassfied datapoints at soft margin
    clf = svm.SVC(kernel='linear', C=1)     # decison boundary is curved => dataset is linearly sepertable
    clf = svm.SVC(kernel='poly', degree=3)  # Third-degree polynomial kernel => decison boundary is curved => dataset is linearly sepertable
    clf = svm.SVC(kernel='rbf')             # default scikit-learn's SVM implementation => dataset is NOT linearly sepertable
    clf = svm.SVC(kernel='sigmoid')         # dataset is NOT linearly sepertable => decision boundary has a sigmoid shape
    To find the best kernel function, needs to turn hyperparameters
    X, y = datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42) # sample datasets

    """
    X = x_dataset
    y = y_label
    clf = svm.SVC(kernel='linear', C=1)     # create the kernel 
    clf.fit(X, y)             # train the model
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 # create d training data but smaller than 1 and greater than 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02)) # create the grid based on above value
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # predict the value
    Z = Z.reshape(xx.shape)

    # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)  # plot the decision boundary
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    # plt.xlabel('X1')
    # plt.ylabel('X2')
    # plt.title('SVM Decision Boundary')
    # plt.show()
    return Z


# decision tree
def dt_classifier(x_dataset,y_label):
    """
    can be used for both regression and classification problems
    data = load_iris()
    x_dataset = data.data
    y_label = data.target
    """
    X = x_dataset
    y = y_label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # Split the dataset into training and testing sets
    clf = DecisionTreeClassifier()  # Create a Decision Tree classifier
    clf.fit(X_train, y_train)       # Train the classifier on the training data
    y_pred = clf.predict(X_test)    # Make predictions on the test data
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# random forest
def rf_classifier(x_dataset,y_label):
    """
    tree refers to one of decision tree from decision tree model
    
    """
    X = x_dataset
    y = y_label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # Split the dataset into training and testing sets
    clf = RandomForestClassifier(n_estimators=100, random_state=42) # Create a Random Forest classifier with 100 trees 
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# cat boost
def catboost_clssifier(x_dataset,y_label):
    """
    data = load_iris()
    x_dataset = data.data
    y_label   = (data.target == 2).astype(int)  # Convert to binary classification
    """
    X = x_dataset
    y = y_label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # Split the dataset into training and testing sets

    model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, loss_function='Logloss', cat_features=[])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report =classification_report(y_test, y_pred)
    return accuracy, report
# light gbm
def lgb_classifier(x_dataset,y_label):
    """
    data = load_iris()
    x_dataset = data.data
    y_label   = (data.target == 2).astype(int)  # Convert to binary classification

    
    """ 
    X = x_dataset
    y = y_label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05}
    num_round = 100  # Number of boosting rounds
    model = lgb.train(params, train_data, num_round)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration) # Make predictions on the test data
    y_pred_binary = (y_pred > 0.5).astype(int) # Convert predicted probabilities to binary predictions
    accuracy = accuracy_score(y_test, y_pred_binary)# Evaluate the model
    return accuracy