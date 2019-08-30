#This code is written by Muntasir FEROZ
#Moisture, temperature and pulse  is taken
#Here only the right chosen algorithm is used which is decision tree
#Classification and Regression Trees (CART).
#Single prediction of random values are
# used to check whether the classifiers are giving right answers
#Lastly have the ability to export the decision tree model as pickel
#that is the model can be deployed for practical application
'''

 Here we will predict whether a user's physical condition is Critical, Moderate
 or Normal depending on the attributes  Moisture, Temperature and Heart rate/Pulse

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
dataset = pd.read_excel('D:\Myworkplace\Python\health_care\dataset\h1022.xlsx', 'Sheet1', index_col=None, na_values=['NA'])



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
#X = X.drop('MOISTURE', axis=1)# the new X variable contains all the columns from the X,
                                # except the "MOISTURE" column, which is the label.

y = dataset['CONDITION']    #The y variable contains the values from the "Class" column.
                        # The X variable is our attribute set and y variable contains corresponding labels.


'''
The final preprocessing step is to divide our data into training and test sets. 
The model_selection library of Scikit-Learn contains train_test_split method, 
which we'll use to randomly split the data into training and testing sets
In the code below, the test_size parameter specifies the ratio of the test set,
 which we use to split up 20% of the data in to the test set and 80% for training.
'''
#EMULATING DO WHILE LOOP SO THAT THE MODEL WITH MOST ACCURACY CAN BE CHOOSEN BY CONTINUOUS ITTERATION
#IF ONE DOES NOT LIKE
while True:
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

    #============================= DOING PREDICTION=========================
    from sklearn.tree import DecisionTreeClassifier

    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
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

    #=================================

    from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
    print("------------------CLASIFIER-------------")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred)*100)

    #==================================== NEW DATA PREDICTION====================
    '''
    #loading data from an excel data sheet and feeding it to the learned model for giving prediction

    dataset1 = pd.read_excel('D:\Myworkplace\Python\health_care\dataset\PLX-DAQ.xlsx', 'Simple Data', index_col=None, na_values=['NA'])

    check_data = dataset1.drop('MOISTURE', axis=1)#X variable contains all the columns from the dataset,
                                # except the "Class" column, which is the label.
    check_data= check_data.drop('TEMPERATURE', axis=1)#X variable contains all the columns from the dataset,
                                # except the "Class" column, which is the label.
    check_data = check_data.drop('PULSE', axis=1)#X variable contains all the columns from the dataset,
                                # except the "Class" column, which is the label.
    print(check_data.head())

    '''

    #print(dataset1.head())
    test_moisture=70
    test_temperature = 102
    test_pulse = 120

    #used to test manually
    #y_pred = classifier.predict(X_test)
    y_pred = classifier.predict([[test_moisture,test_temperature,test_pulse]])

    #used to check data which is loaded from the second excel sheet
    #y_pred=classifier.predict(check_data)

    print('Prediction Decision Tree Classifier:')
    print(y_pred)

    #=====================================================================================
    #***********************************EXPORTING THE LEARNED MODEL***************************
    #=======================================================================================

    '''
    EXPORTING THE MODEL FOR FUTURE USE
    using pickle package the learned model can be serialized into a form so that the learned
    model can be deployed practically. Using pickle package one have to train the model once.      
    '''
    import pickle #for serializing the learned model
    yes=input("DO You Want Serialize The Learned Model?(y/n)")
    if(yes=='y'or yes=='Y'):

        filename=input("Give a file name") #giving the learned model a file name
        print(filename + '.sav')
        pickle.dump(classifier,open('Serialize_Models\ ' + filename + '.sav', 'wb'))#saving the model in Serialize_Models directory
        print("The Model Have Been Serialized")                                     #that is saving the model in disk

    else:
        print("The Model Have Not Been Serialized")

    #EMULATING DO WHILE LOOP
    yes_exit=input("DO you want to exit the program(y/n)")
    if(yes_exit=='y' or yes_exit == 'Y'):#if this is true then the loop breaks
        break #emulating do while loop

