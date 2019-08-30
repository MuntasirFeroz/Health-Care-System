#This code is written by Muntasir FEROZ
#Here Decision tree classifier and Decision tree classifier using GINI is used
#
#
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
#*************************************************DATA ANALYSIS****************************************
#------------------------------------------------------------------------------------------------------

print(dataset.describe())#to see statistical details of the dataset
print(dataset.shape)   #to see the number of rows and columns in our dataset
print(dataset.head())   #to see the first five records of the dataset


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
y = dataset['CONDITION']    #The y variable contains the values from the "Class" column.
                        # The X variable is our attribute set and y variable contains corresponding labels.

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
#print(X_test[0:2])
#y_pred = classifier.predict(X_test[0:1])
#print('Prediction :')
#print(y_pred)
y_pred = classifier.predict(X_test)
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


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



