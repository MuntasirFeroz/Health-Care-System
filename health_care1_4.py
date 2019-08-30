#This code is written by Muntasir FEROZ
#here only taking temperature and pulse. Moisture is excluded(will be changed toincluded)
#Single prediction of random values are
# used to check whether the classifiers are giving right answers
#Here Decision tree classifier and Decision tree classifier using GINI is used
#AND ALSO 6 different algorithms:
#Logistic Regression (LR)
#Linear Discriminant Analysis (LDA)
#K-Nearest Neighbors (KNN).
#Classification and Regression Trees (CART).
#aussian Naive Bayes (NB).
#Support Vector Machines (SVM).
#GRAPH ARE USED TO VIZUALZE THE DATASET
#Bar graph is used here for comparing the differnt algorithms

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

Note: Moisture was excluded due less co-relation with data, so here only temperature and Pulse rate is used
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

'''
We don’t know which algorithms would be good on this problem or what configurations to use. 
Let’s evaluate 6 different algorithms:

Logistic Regression (LR)
Linear Discriminant Analysis (LDA)
K-Nearest Neighbors (KNN).
Classification and Regression Trees (CART).
Gaussian Naive Bayes (NB).
Support Vector Machines (SVM).

This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms.
We reset the random number seed before each run to ensure that the evaluation of each algorithm is
performed using exactly the same data splits. It ensures the results are directly comparable.
'''
#==================
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
#==================

'''
We now have 6 models and accuracy estimations for each. 
We need to compare the models to each other and select the most accurate.
We can also create a plot of the model evaluation results and compare the spread
 and the mean accuracy of each model. There is a population of accuracy measures 
 for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation).
'''
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


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
#============================= DOING PREDICTION=========================

#-------Decision Tree (CART)---------------

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#-----logistic Regression (LB)---------
logisticRegression_classifier = LogisticRegression(solver='liblinear', multi_class='ovr')
logisticRegression_classifier.fit(X_train, y_train)
y_pred_logisticRegression=logisticRegression_classifier.predict(X_test)
#-----Linear Discriminant Analysis-----------
linearDiscriminantAnalysis_classifier = LinearDiscriminantAnalysis()
linearDiscriminantAnalysis_classifier.fit(X_train, y_train)
y_pred_linearDiscriminantAnalysis=linearDiscriminantAnalysis_classifier.predict(X_test)
#---------K Neighbors Classifier---------------
kNeighbors_classifier = KNeighborsClassifier()
kNeighbors_classifier.fit(X_train, y_train)
y_pred_kNeighbors=kNeighbors_classifier.predict(X_test)
#--------------GaussianNB----------------------
gaussianNB_classifier = GaussianNB()
gaussianNB_classifier.fit(X_train, y_train)
y_pred_gaussianNB=gaussianNB_classifier.predict(X_test)
#-------------------SVM-------------------------
sVM_classifier = SVC(gamma='auto')
sVM_classifier.fit(X_train, y_train)
y_pred_sVM=sVM_classifier.predict(X_test)

#=====================================================================================
#***********************************EVALUATING THE ALGORITHM***************************
#=======================================================================================

'''
Since we have trained our algorithm and made some predictions.
 Now we'll see how accurate our algorithm is?  For classification tasks some 
 commonly used metrics are confusion matrix, precision, recall, and F1 score. 
 It is good for us Scikit=-Learn's metrics library contains the classification_report 
 and confusion_matrix methods that can be used to calculate these metrics for us
'''

#=================================

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,average_precision_score,precision_score,f1_score,roc_auc_score

print("-------------Decision Tree (CART)-------------")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Acurracy score: ")
d_accuracy=accuracy_score(y_test, y_pred)*100
print(d_accuracy)
print("Precision score: ")
d_precision=precision_score(y_test, y_pred,average='macro')*100
print(d_precision)
print("F1 score: ")
d_f1=f1_score(y_test, y_pred,average='macro')*100
print(d_f1)

print("-------------Logistic Regression-------------")
print(confusion_matrix(y_test, y_pred_logisticRegression))
print(classification_report(y_test, y_pred_logisticRegression))
print("Acurracy score: ")
lr_accuracy=accuracy_score(y_test, y_pred_logisticRegression)*100
print(lr_accuracy)

print("Precision score: ")
#lr_precision=(precision_score(y_test, y_pred_logisticRegression,average='weighted')*100)
lr_precision=precision_score(y_test, y_pred_logisticRegression,average='macro')
print(lr_precision)

print("F1 score: ")
lr_f1=(f1_score(y_test, y_pred_logisticRegression,average='macro')*100)
print(lr_f1)
print("-------------Linear Discriminant Analysis-------------")
print(confusion_matrix(y_test, y_pred_linearDiscriminantAnalysis))
print(classification_report(y_test, y_pred_linearDiscriminantAnalysis))
print("Acurracy score: ")
li_accuracy=accuracy_score(y_test, y_pred_linearDiscriminantAnalysis)*100
print(li_accuracy)
print("Precision score: ")
li_precision= precision_score(y_test, y_pred_linearDiscriminantAnalysis,average='macro')*100
print(li_precision)
print("F1 score: ")
li_f1=f1_score(y_test, y_pred_linearDiscriminantAnalysis,average='macro')*100
print(li_f1)
print("-------------K Neighbors Classifier-------------")
print(confusion_matrix(y_test, y_pred_kNeighbors))
print(classification_report(y_test, y_pred_kNeighbors))
print("Acurracy score: ")
kn_accuracy=accuracy_score(y_test, y_pred_kNeighbors)*100
print(kn_accuracy)
print("Precision score: ")
kn_precision=precision_score(y_test, y_pred_kNeighbors,average='weighted')*100
print(kn_precision)
print("F1 score: ")
kn_f1=f1_score(y_test, y_pred_kNeighbors,average='weighted')*100
print(kn_f1)
print("-------------GaussianNB-------------")
print(confusion_matrix(y_test, y_pred_gaussianNB))
print(classification_report(y_test, y_pred_gaussianNB))
print("Acurracy score: ")
g_accuracy=accuracy_score(y_test, y_pred_gaussianNB)*100
print(g_accuracy)
print("Precision score: ")
g_precision=precision_score(y_test, y_pred_gaussianNB,average='weighted')*100
print(g_precision)
print("F1 score: ")
g_f1=f1_score(y_test, y_pred_gaussianNB,average='weighted')*100
print(g_f1)
print("-------------SVM-------------")
print(confusion_matrix(y_test, y_pred_sVM))
print(classification_report(y_test, y_pred_sVM))
print("Acurracy score: ")
s_accuracy=accuracy_score(y_test, y_pred_sVM)*100
print(s_accuracy)
print("Precision score: ")
s_precision=precision_score(y_test, y_pred_sVM,average='weighted')*100
print(s_precision)
print("F1 score: ")
s_f1=f1_score(y_test, y_pred_sVM,average='weighted')*100
print(s_f1)
#======================================CREATING A BAR GRAPH TO COMPARE ALGOS=======================
#======================================WITH ACCURACY, PRECISION, F1 SCORE=============================

# data to plot
n_groups = 6# accuracy,prcision,f1-score

means_accuracy = (d_accuracy ,lr_accuracy,li_accuracy,kn_accuracy,s_accuracy,g_accuracy)
means_precision=(d_precision,lr_precision,li_precision,kn_precision,s_precision,g_precision)
means_f1_score = (d_f1,lr_f1,li_f1,kn_f1,s_f1,g_f1)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8

rects1 = plt.bar(index, means_accuracy, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Accuracy')

rects2 = plt.bar(index + bar_width, means_precision, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Precision')
rects3 = plt.bar(index + bar_width*2, means_f1_score, bar_width,
                 alpha=opacity,
                 color='r',
                 label='F1 Score')


plt.xlabel('Algorithms')
plt.ylabel('Percentage')
plt.title('Comparison between Algorithms')
plt.xticks(index + bar_width, ('CART', 'LR', 'LDA', 'KNN','SVM','NB'))
plt.legend()

plt.tight_layout()
plt.show()


#==================================== NEW DATA PREDICTION================================================

#to test for individual cases
test_temperature = 102
test_pulse = 120


#y_pred = classifier.predict(X_test)
y_pred = classifier.predict([[test_temperature,test_pulse]])
print('Prediction Decision Tree Classifier:')
print(y_pred)


y_pred_LR = logisticRegression_classifier.predict([[test_temperature,test_pulse]])
print('Prediction Logistic Regression Classifier:')
print(y_pred_LR)

y_pred_LDA = linearDiscriminantAnalysis_classifier.predict([[test_temperature,test_pulse]])
print('Prediction Linear DiscriminantAnalysis Classifier:')
print(y_pred_LDA)

y_pred_KNN = kNeighbors_classifier.predict([[test_temperature,test_pulse]])
print('Prediction kNeighbors Classifier:')
print(y_pred_KNN)

y_pred_NB = gaussianNB_classifier.predict([[test_temperature,test_pulse]])
print('Prediction gaussianNB Classifier:')
print(y_pred_NB)

y_pred_svm = sVM_classifier.predict([[test_temperature,test_pulse]])
print('Prediction SVM Classifier:')
print(y_pred_svm)


