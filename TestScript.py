import pickle
from Data_Handling  import *
from Regression import  *
from  Classification import  *
import pandas as pd
'''
#------- Regression-------------------
dataset =Preprocessing_Our_Data("Regression",pd.read_csv('CarPrice_training.csv'),'Drop All Null Rows',True,False,True,False)
X_train_D, X_test_D, y_train_D, y_test_D = Split_Data(dataset,0.2,1,0,False,False)

#X_test =dataset[:,:-1]
#y_test= dataset[:,-1]

# Load from file
pkl_filename='Linear_Model.pkl'
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

# Calculate the accuracy score and predict target values
score = pickle_model.score(X_test, y_test)
print("Test score: {0:.2f} %".format(100 * score))
#Ypredict = pickle_model.predict(X)
'''
#------- Classification------------------- KNN--Logistic Regression
dataset = Preprocessing_Our_Data("Classification",pd.read_csv('CarPrice_training_classification.csv'),"Mean",False,True,True,False)
X_train, X_test, y_train, y_test = Split_Data(dataset,0.2,2,True,'','')

#X_test =dataset[:,:-1]
#y_test= dataset[:,-1]

# Load from file
pkl_filename='Knn_Model.pkl'
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

# Calculate the accuracy score and predict target values
score = pickle_model.score(X_test, y_test)
print("Test score: {0:.2f} %".format(100 * score))
#Ypredict = pickle_model.predict(X)