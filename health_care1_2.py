#This code is written by Muntasir FEROZ
#here only taking temperature and pulse
#GRAPH ARE USED TO VIZUALZE THE DATASET
#Single prediction of random values are
# used to check whether the classifiers are giving right answers
#Here Decision tree classifier and Decision tree classifier using GINI is used

'''

 Here we will predict whether a user's physical condition is Critical, Moderate
 or Normal depending on the attributes  Moisture, Temperature and Heart rate/Pulse
 Here,
 Moderate is represented by -1
 Normal is represented by 0
 Critical is represented by 1

'''

#***************************************************IMPORTING LIBRARIES***************************
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
#%matplotlib inline
#----------------------
from IPython import get_ipython


#**************************************************IMPORTING DATASETS****************************

'''Since the file is in CSV format, we will use panda's read_csv method to read our CSV data file.
#If the file is in xlsx format then read_excel method can be used'''
dataset = pd.read_excel('D:\Myworkplace\Python\health_care\dataset\health_dataset.xlsx', 'Sheet1', index_col=None, na_values=['NA'])

#dataset = pd.read_csv('D:\Myworkplace\Python\hfake_bank\dataset\data_bill_authentication.csv')

#-----------------------------------------------------------------------------------------------------
#*************************************************DATA ANALYSIS AND VISUALIZATION****************************************
#------------------------------------------------------------------------------------------------------

print(dataset.describe())#to see statistical details of the dataset
print(dataset.shape)   #to see the number of rows and columns in our dataset
print(dataset.groupby('CONDITION').size()) #to see number of instances for each class
print(dataset.head())   #to see the first five records of the dataset

'''
We are going to look at two types of plots:

    1)Univariate plots to better understand each attribute.
    2)Multivariate plots to better understand the relationships between attributes.
'''
#---------------------------UNIVARIET PLOT------------------------------
#We start with some univariate plots, that is, plots of each individual variable.
# box and whisker plots
#gives us a much clearer idea of the distribution of the input attributes

dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#We can also create a histogram of each input variable to get an idea of the distribution.
# histograms
dataset.hist()
plt.show()
#-------------------------MULTIVARIATE PLOTS--------------------
'''
Now we can look at the interactions between the variables.
First, we will look at scatterplots of all pairs of attributes. 
This can be helpful to spot structured relationships between input variables.
Note the diagonal grouping of some pairs of attributes. 
This suggests a high correlation and a predictable relationship.
'''
# scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(dataset)
plt.show()

#========================================================================================================
#************************************************PREPARING THE DATASET***********************************
#========================================================================================================
'''
we will divide our data into attributes and labels and will then divide the resultant data into 
both training and test sets. By doing this we can train our algorithm on one set of data and then 
test it out on a completely different set of data that the algorithm hasn't seen yet. 
This provides you with a more accurate view of how your trained algorithm will actually perform.
'''


X = dataset.drop('CONDITION', axis=1)#X variable contains all the columns from the dataset,
                                # except the "Class" column, which is the label.
X = X.drop('MOISTURE', axis=1)# the new X variable contains all the columns from the X,
                                # except the "MOISTURE" column, which is the label.

y = dataset['CONDITION']    #The y variable contains the values from the "Class" column.
                        # The X variable is our attribute set and y variable contains corresponding labels.

print('PRINTING X---------')
print(X.head())

'''
The final preprocessing step is to divide our data into training and test sets. 
The model_selection library of Scikit-Learn contains train_test_split method, 
which we'll use to randomly split the data into training and testing sets
In the code below, the test_size parameter specifies the ratio of the test set,
 which we use to split up 20% of the data in to the test set and 80% for training.
'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


#=====================================================================================================
#*************************************TRAINING AND MAKING PREDICTIONS**********************************
#=====================================================================================================

'''
Since the data has been divided into the training and testing sets, 
the final step is to train the decision tree algorithm on this data 
and make predictions. Scikit-Learn contains the tree library,
which contains built-in classes or methods for various decision tree algorithms. 
Since we are going to perform a classification task here, we will use 
the DecisionTreeClassifier class for this example. The fit method of this 
class is called to train the algorithm on the training data, which is passed 
as parameter to the fit method.
'''
#==================
from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=3)
clf_gini.fit(X_train, y_train)

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')
#=====================

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

'''
Now that our classifier has been trained, let's make predictions on the test data. 
To make predictions, the predict method of the DecisionTreeClassifier class is used.
'''

'''
print("=============DATA=========")
print(X_test[0:1])
print(y_test[0:1])
y_pred_gini=clf_gini.predict(X_test[0:1])
print('Prediction GINI:')
print(y_pred_gini)

y_pred = classifier.predict(X_test[0:1])
print('Prediction Classifier:')
print(y_pred)

'''
y_pred_gini=clf_gini.predict(X_test)
#y_pred_gini=clf_gini.predict([[89,12]])
#print('Prediction GINI:')
#print(y_pred_gini)

y_pred = classifier.predict(X_test)
#y_pred = classifier.predict([[89,12]])
#print('Prediction Classifier:')
#print(y_pred)

#y_pred = classifier.predict(X_test)

#=====================================================================================
#***********************************EVALUATING THE ALGORITHM***************************
#=======================================================================================

'''
Since we have trained our algorithm and made some predictions.
 Now we'll see how accurate our algorithm is?  For classification tasks some 
 commonly used metrics are confusion matrix, precision, recall, and F1 score. 
 Lucky for us Scikit=-Learn's metrics library contains the classification_report 
 and confusion_matrix methods that can be used to calculate these metrics for us
'''


from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print("------------------GINI-------------")
print(confusion_matrix(y_test, y_pred_gini))
print(classification_report(y_test, y_pred_gini))
print(accuracy_score(y_test, y_pred_gini)*100)

print("------------------CLASIFIER-------------")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred)*100)

#==================================== NEW DATA PREDICTION====================
#y_pred_gini=clf_gini.predict(X_test)
y_pred_gini=clf_gini.predict([[105,70]])
print('Prediction GINI:')
print(y_pred_gini)

#y_pred = classifier.predict(X_test)
y_pred = classifier.predict([[105,70]])
print('Prediction Classifier:')
print(y_pred)
