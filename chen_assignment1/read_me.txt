Command Line: python homework_1_programming.py

Library used:
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
%matplotlib inline
import math

Problem 1:

I used pandas library to read the csv files that the instructors uploaded on Blackboard.

Problem 2:

I used matplotlib library to display the relationship between any two features in the dataset. You can input any two features in the dataset into the feature_display() function. 

Problem 3:

I used merge sort to sort each feature value from the smallest to the largest and determine which feature can better separate the classes.

Here are the steps that I used to develop the algorithm:
1) Use the groupby function to check the average Sepal.Length, Sepal.Width, Petal.Length and Petal.Width. Create a dictionary to rank the species based on its Sepal.Length, Sepal.Width, Petal.Length and Petal.Width. The key of the dictionary is the plant's feature and the key value is the species ranking from smallest to the largest based on their average feature values.
2) Create a merge sort function that can sort a list from smallest to largest. Recuisively call the merge sort() function on the list until it only has one element, then, we use merge() function to merge the sorted lists. 
I used the reference from here: https://www.youtube.com/watch?v=_trEkEX_-2Q
3) Create a list to store the species values and repeat each species value based on their occurrences in the original dataset
4) Convert the sorted feature and species list into pandas columns, create a dataframe that store the values from the sorting algorithms
5) Gather the values from the original dataset
6) Compare the values in these two dataframes to get the accuracy of the sorting algorithm in determining its species

As we can see from the result, the merge sort algorithm with Petal Length as its feature is the most effective in separating the species, with an accuracy of 96%. 