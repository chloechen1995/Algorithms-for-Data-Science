#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# ### Step 1: Data Cleansing
# Fill in the nan value with the average feature value per class

# In[2]:


iris_clean = pd.read_csv('iris_data_for_cleansing.csv')


# In[3]:


# find the rows with nan values
iris_clean[iris_clean.isnull().any(axis = 1)]


# In[4]:


feature_list = ['sepal length', 'sepal width', 'petal length', 'petal width',
       'New Feature 1', 'New Feature 2']


# In[5]:


# fill in the nan value with the average feature value per class
for feature in feature_list:
    iris_clean[feature] = iris_clean.groupby('class')[feature].transform(lambda x: x.fillna(x.mean()))


# In[6]:


iris_clean[iris_clean.isnull().any(axis = 1)]


# ### Step 2: Data Transformation
# Use PCA from the scikit-learn library and successfully used three principal components to capture the essence of the data. It explained 95.9% of the data.

# In[7]:


x = iris_clean.loc[:, 'sepal length': 'New Feature 2']
x = StandardScaler().fit_transform(x)
pca = PCA(n_components = 3)
principalComponents = pca.fit_transform(x)
principal_df = pd.DataFrame(principalComponents, columns = ['principal_component_1', 'principal_component_2', 'principal_component_3'])
final_df = pd.merge(principal_df, iris_clean[['class']], left_index=True, right_index=True)
pca.explained_variance_ratio_.sum()


# In[ ]:





# ### Step 3: Generate two sets of features from the original 4 features
# 1) Generate covariance matrix for the specified species using the two features entered
# 
# 2) Create an array to store the average feature value for each species
# 
# 3) Draw random samples from the multivaiate normal distribution using the covariance matrix in step 1 and the mean array in step 2
# 
# 4) Limit the values for samples generated from distribution so that it does not exceed the maximum value or fall below the minimum value in the original dataset

# In[8]:


def gather_cov_matrix_class(Species, dataframe, feature_1, feature_2):
    cov_matrix_class = dataframe[dataframe['class'] == Species][[feature_1, feature_2]].cov()
    cov_array_class = np.array(cov_matrix_class)
    
    average_list = []
    feature_list = [feature_1, feature_2]
    
    for feature in feature_list:
        average_feature = np.mean(dataframe[dataframe['class'] == Species][feature])
        average_list.append(average_feature)
    
    average_array = np.array(average_list)
    
    new_array = np.random.multivariate_normal(average_array, cov_array_class, 50)
    
    final_df = pd.DataFrame(new_array, columns = [feature_1, feature_2])
    
    max_dict = dataframe[dataframe['class'] == Species][[feature_1, feature_2]].max()
    
    min_dict = dataframe[dataframe['class'] == Species][[feature_1, feature_2]].min()
    
    new_df = pd.DataFrame()
    
    for feature in feature_list:
        new_df = pd.concat([new_df, np.clip(final_df[feature], min_dict[feature], max_dict[feature])], axis = 1)
    
    new_df.columns = ['generated_feature_1', 'generated_feature_2']
    
    return new_df


# In[9]:


output_final_df = pd.DataFrame()

for species in iris_clean['class'].unique():
    output_final_df = pd.concat([output_final_df, gather_cov_matrix_class(species, iris_clean, 'sepal width', 'petal width')])


# In[10]:


output_final_df.head()


# ### Step 4: Perform Feature Preprocessing
# Reference: https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba
# 
# (1) Use Z score to detect the outliers
# 
# (2) If the Z score is greater than 3, the value is an outlier

# In[11]:


new_iris = pd.concat([iris_clean, output_final_df.reset_index(drop = True)], axis = 1)


# In[12]:


new_iris.head()


# In[13]:


feature_list = ['sepal length', 'sepal width', 'petal length', 'petal width',
       'New Feature 1', 'New Feature 2', 'generated_feature_1',
       'generated_feature_2']


# In[14]:


# check whether there is outlier for each feature in each class
index_list = []
for species in iris_clean['class'].unique():
    for feature in feature_list:
        z = np.abs(stats.zscore(new_iris[new_iris['class'] == species][feature]))
        output_array = np.where(z > 3)
        if len(*output_array) > 0:
            for i in range(len(output_array)):
                index_list.append(*output_array[i])
        


# In[15]:


for index in index_list:
    final_iris = new_iris.drop(new_iris.index[index])


# In[16]:


final_iris.shape


# ### Step 5: Feature ranking
# Reference: https://scikit-learn.org/stable/modules/feature_selection.html
# 
# Use the univariate feature selection model provided by Scikit Learn.
# 
# Found petal length and petal width are top two features. 

# In[17]:


# feature matrix
X = iris_clean.loc[:, 'sepal length': 'New Feature 2']


# In[18]:


y = iris_clean['class']


# In[19]:


X.head()


# In[20]:


X_best_two = SelectKBest(chi2, k = 2).fit_transform(X, y)


# In[21]:


X_best_two


# ### Step 6: Dimension Reduction
# Reduce the dimensionality to two features and they captures 82.3% of the data. 

# In[22]:


x = iris_clean.loc[:, 'sepal length': 'New Feature 2']
x = StandardScaler().fit_transform(x)
pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(x)
principal_df = pd.DataFrame(principalComponents, columns = ['principal_component_1', 'principal_component_2'])
final_df_principal = pd.merge(principal_df, iris_clean[['class']], left_index=True, right_index=True)
pca.explained_variance_ratio_.sum()


# ### Step 7: Machine Learning Techniques

# (a) Expectation Maximization
# 
# Reference: Professor's Expectation Maximization Powerpoint presentation
# 
# (1) Create array to store the feature's mean and standard deviation
# 
# (2) Generate the conditional probability in the expectation step, computes the expected value of Xn data using the current estimation of the parameter and the observed data
# 
# (3) Keep updating the means, standard deviations and mixing probabilities in the maximization step
# 

# In[24]:


def create_df(df):
    mean_list = []
    std_list = []
    for col in df.columns:
        mean_list.append(df[col].mean())
        std_list.append(df[col].std())
        
    mean_array = np.array(mean_list)
    
    std_array = np.array(std_list)
    
    x_matrix = np.array(df.values)
    
    return mean_array, std_array, x_matrix


# In[25]:


def expectation_mean_std_matrix(df, mean_array, std_array, random_matrix, initialize_array):
    left_matrix = np.matmul(mean_array.reshape(-1, 1), initialize_array.reshape(1, -1))
    right_matrix = np.matmul(std_array.reshape(-1, 1), random_matrix.reshape(1, -1))
    mean_matrix = np.add(left_matrix, right_matrix)
    average_std = np.array(std_array.mean())
    std_matrix = np.matmul(average_std.reshape(1, 1), initialize_array.reshape(1, -1))
    return(mean_matrix, std_matrix)


# In[26]:


def expectation_prob(k, initialize_array, df, mean_matrix, std_matrix, x_matrix):
    prob_k_n = initialize_array.reshape(1, -1) / k
    
    final_list = []
    for i in range(mean_matrix.shape[1]):
        mean_array_new = np.repeat([mean_matrix[:, i]], x_matrix.shape[0], axis = 0)

        norm_array = np.subtract(x_matrix, mean_array_new)
        
        left = 1/(math.sqrt(2*math.pi)* std_matrix[0][i])**2 * np.exp((-1/2))
        
        output_list = left * LA.norm(norm_array, ord = 2, axis = 1)**2 / std_matrix[0][i] ** 2

        final_list.append(output_list)
    
    final_array = np.array(final_list)
    
    sum_p_k_g = np.matmul(prob_k_n, final_array)
    
    mean_list_output = []

    for i in range(prob_k_n.shape[1]):
        mean_list_output.append(prob_k_n[:, i] * final_array[i, ]/sum_p_k_g)
    
    prob_k_n_matrix = np.array(mean_list_output)
    
    return prob_k_n_matrix
        


# In[27]:


def maximization_mean_std_matrix(prob_k_n_matrix, x_matrix, d, N, k):
    sum_matrix = np.sum(prob_k_n_matrix.reshape(prob_k_n_matrix.shape[0], prob_k_n_matrix.shape[-1]), axis = 1)
    final_sum = np.repeat([sum_matrix], x_matrix.shape[-1], axis = 0)
    mean_array = np.matmul(prob_k_n_matrix, x_matrix).transpose().reshape(x_matrix.shape[-1], -1)/final_sum

    output_list = []
    for i in range(k):
        x_m_array = np.repeat([mean_array[:, i]], x_matrix.shape[0], axis = 0)
        norm_array_new = np.subtract(x_matrix, x_m_array)
        left_array = prob_k_n_matrix.reshape(-1, x_matrix.shape[0])[i].reshape(x_matrix.shape[0], 1)
        sigma_array = left_array * (norm_array_new ** 2)
        output_list.append(np.sqrt(np.sum(sigma_array, axis = 1).sum()/ (d * sum_matrix[i])))

    sigma_array_new = np.array([output_list])

    new_prob = (1/N) * sum_matrix.reshape(k, -1)

    return mean_array, sigma_array_new, new_prob


# In[28]:


def EM(df, random_matrix, initialize_array, k, d, N):
    mean_array, std_array, x_matrix = create_df(df)
    mean_matrix, std_matrix = expectation_mean_std_matrix(df, mean_array, std_array, random_matrix, initialize_array)
    prob_k_n_matrix = expectation_prob(k, initialize_array, df, mean_matrix, std_matrix, x_matrix)
    
    output_dict = {'iteration_1': [mean_matrix, std_matrix]}
    for i in range(1, 6):
        prob_k_n_matrix = expectation_prob(k, initialize_array, df, output_dict['iteration_' + str(i)][0], 
                         output_dict['iteration_' + str(i)][1], x_matrix)
        output_dict['iteration_' + str(i)].append(prob_k_n_matrix)
        mean_updated_array, std_updated_array, new_prob = maximization_mean_std_matrix(prob_k_n_matrix, x_matrix, d, N, k)
        output_dict.update({'iteration_' + str(i + 1) : [mean_updated_array, std_updated_array]})
        output_dict['iteration_' + str(i + 1)][0] = mean_updated_array
        output_dict['iteration_' + str(i + 1)][1] = std_updated_array
        output_dict['iteration_' + str(i)].append(new_prob)
        
    return output_dict


# In[29]:


initialize_array = np.array([1,1,1])
random_matrix = np.array([-0.18671, 0.72579, 0.52579])
k = 3
d = 2
N = final_df.shape[0]


# In[30]:


iris = final_df_principal.loc[:, 'principal_component_1' : 'principal_component_2']


# In[31]:


EM(iris, random_matrix, initialize_array, k, d, N)


# (b) Fisher Linear Discriminant
# 
# Reference: https://sebastianraschka.com/Articles/2014_python_lda.html
# 
# (1) Create a list to store the species and plant's features
# 
# (2) Create a vector to store the average plant's feature vallue for each species
# 
# (3) Calculate the within-class scatter matrix
# 
# (4) Calculate the between-class scatter matrix
# 
# (5) Gather the eigenvalues and eigenvectors
# 
# (6) Order the eigenvalues from largest to smallest and get the eigenvectors with relatively large eigenvalues
# 
# (7) Create the eigenvector matrix and calculate its dot product with the original data values

# In[32]:


species_list = [1, 2, 3]


# In[33]:


feature_list = ['principal_component_1', 'principal_component_2']


# In[34]:


final_vector = []
for s in species_list:
    mean_vectors = []
    for f in feature_list:
        mean_vectors.append(final_df_principal[final_df_principal['class'] == s][f].mean())
    final_vector.append(mean_vectors)


# In[35]:


output_vector = [np.array(x) for x in final_vector]


# In[36]:


# calculate the within-class scatter matrix

within_class_matrix = np.zeros((2, 2))

for s, i in zip(species_list, output_vector):
    class_matrix = np.zeros((2, 2))
    for row in final_df_principal[final_df_principal['class'] == s][['principal_component_1', 'principal_component_2']].values:
        x_i = row.reshape(2, 1)
        m_i = i.reshape(2, 1)
        class_matrix = class_matrix + (x_i - m_i).dot((x_i - m_i).T)
    within_class_matrix = within_class_matrix + class_matrix
    


# In[37]:


# calculate the between-class scatter matrix

feature_mean = [final_df_principal[f].mean() for f in feature_list]


# In[38]:


between_class_matrix = np.zeros((2, 2))
for i, s in zip(output_vector, species_list):
    m_i = i.reshape(2, 1)
    final_feature_mean = np.array(feature_mean).reshape(2, 1)
    between_class_matrix = between_class_matrix + len(final_df_principal[final_df_principal['class'] == s]) * (m_i - final_feature_mean).dot((m_i - final_feature_mean).T)


# In[39]:


# gather the eigenvalues and eigenvectors
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(within_class_matrix).dot(between_class_matrix))


# In[40]:


eig_dict = dict(zip(eig_vals, eig_vecs))


# In[41]:


# get the eigenvectors with the largest eigenvalues

eig_dict[max(eig_dict.keys())]


# In[42]:


key_list = sorted(eig_dict, reverse = True)[:2]


# In[43]:


eig_matrix = np.hstack((eig_dict[key_list[0]].reshape(2, 1), eig_dict[key_list[1]].reshape(2, 1)))


# In[44]:


data_val = final_df_principal[['principal_component_1', 'principal_component_2']].values


# In[45]:


data_lda = data_val.dot(eig_matrix)


# (c) Feed Forward Neural Network
# 
# Reference: https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/
# 
# (1) Preprocess the data
# 
# (2) Create the training set and test set 
# 
# (3) Scale the data so that they can be uniformly evaluated
# 
# (4) Apply the neural network in the scikit learn library on the training set and test it on the test set

# In[46]:


X = final_df_principal.loc[:, 'principal_component_1': 'principal_component_2']


# In[47]:


y = final_df_principal.loc[:, 'class']


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# In[49]:


scaler = StandardScaler()


# In[50]:


scaler.fit(X_train)


# In[51]:


X_train = scaler.transform(X_train)


# In[52]:


X_test = scaler.transform(X_test)


# In[53]:


mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8), max_iter = 1000)


# In[54]:


mlp.fit(X_train, y_train)


# In[55]:


predictions = mlp.predict(X_test)


# In[56]:


print(classification_report(y_test, predictions))


# (d) Support Vector Machine
# 
# (1) Apply the support vector machine in scikit learn library on the training set and test it on the test set
# 

# In[57]:


clf = svm.SVC(gamma = 'scale', decision_function_shape='ovo')


# In[58]:


clf.fit(X_train, y_train)


# In[59]:


y_pred = clf.predict(X_test)


# In[60]:


print(classification_report(y_test, y_pred))


# In[ ]:




