Command Line: python homework_2_programming.py

Library used:
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import math
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm

Step 1) I filled in the nan value with the average feature value per class.

Step 2) I used PCA from the scikit-learn library and successfully used three principal components to capture the essence of the data. It explained 95.9% of the data.

Step 3) I used the following steps to generate two sets of features from the original 4 features.

1) Generate covariance matrix for the specified species using the two features entered

2) Create an array to store the average feature value for each species

3) Draw random samples from the multivaiate normal distribution using the covariance matrix in step 1 and the mean array in step 2

4) Limit the values for samples generated from distribution so that it does not exceed the maximum value or fall below the minimum value in the original dataset

Step 4) I used the Z score to detect the outliers. If Z score is greater than 3, the value is an outlier.

Step 5) I used the univariate feature selection model provided by Scikit-Learn library to find that petal length and petal width are top two features. 

Step 6) I reduced the dimensionality to two features and they captures 82.3% of the data. 

Step 7) Machine Learning Techniques

(a) Expectation Maximization

Reference: Professor's Expectation Maximization Powerpoint presentation

(1) Create array to store the feature's mean and standard deviation

(2) Generate the conditional probability in the expectation step, computes the expected value of Xn data using the current estimation of the parameter and the observed data

(3) Keep updating the means, standard deviations and mixing probabilities in the maximization step

(b) Fisher Linear Discriminant

Reference: https://sebastianraschka.com/Articles/2014_python_lda.html

(1) Create a list to store the species and plant's features

(2) Create a vector to store the average plant's feature vallue for each species

(3) Calculate the within-class scatter matrix

(4) Calculate the between-class scatter matrix

(5) Gather the eigenvalues and eigenvectors

(6) Order the eigenvalues from largest to smallest and get the eigenvectors with relatively large eigenvalues

(7) Create the eigenvector matrix and calculate its dot product with the original data values


(c) Feed Forward Neural Network

Reference: https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/

(1) Preprocess the data

(2) Create the training set and test set 

(3) Scale the data so that they can be uniformly evaluated

(4) Apply the neural network in the scikit learn library on the training set and test it on the test set


(d) Support Vector Machine

(1) Apply the support vector machine in scikit learn library on the training set and test it on the test set
