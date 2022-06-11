import Data_Handling,Regression
from tkinter import *
from Data_Handling import *
from Regression import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle

print('-----------------------Linear ----------------------------------------')
dataset =Preprocessing_Our_Data("Regression",pd.read_csv('CarPrice_training.csv'),'Drop All Null Rows',True,False,True,False)
X_train_D, X_test_D, y_train_D, y_test_D = Split_Data(dataset,0.2,1,0,False,False)
r2score,train_time,model,Mean_Error,Train_accuracy,Test_accuracy= OUR_Linear(X_train_D, X_test_D, y_train_D, y_test_D)
print("Train Time = ",train_time)
print("R2_Score = ",round(r2score, 2))
print("Mse = ",round(Mean_Error, 2))
print("Train Acc = ",round(Train_accuracy, 2))
print("Test Acc = ",round(Test_accuracy, 2))

print('-----------------------Lasso ----------------------------------------')
dataset =Preprocessing_Our_Data("Regression",pd.read_csv('CarPrice_training.csv'),'Drop All Null Rows',True,False,True,False)
X_train_D, X_test_D, y_train_D, y_test_D = Split_Data(dataset,0.2,1,0,False,False)
r2score,train_time,model,Mean_Error,Train_accuracy,Test_accuracy= OUR_Lasso(X_train_D, X_test_D, y_train_D, y_test_D)
print("Train Time = ",train_time)
print("R2_Score = ",round(r2score, 2))
print("Mse = ",round(Mean_Error, 2))
print("Train Acc = ",round(Train_accuracy, 2))
print("Test Acc = ",round(Test_accuracy, 2))

print('-----------------------Ridge ----------------------------------------')
dataset =Preprocessing_Our_Data("Regression",pd.read_csv('CarPrice_training.csv'),'Drop All Null Rows',True,False,True,False)
X_train_D, X_test_D, y_train_D, y_test_D = Split_Data(dataset,0.2,1,0,False,False)
r2score,train_time,model,Mean_Error,Train_accuracy,Test_accuracy= OUR_Ridge(X_train_D, X_test_D, y_train_D, y_test_D)
print("Train Time = ",train_time)
print("R2_Score = ",round(r2score, 2))
print("Mse = ",round(Mean_Error, 2))
print("Train Acc = ",round(Train_accuracy, 2))
print("Test Acc = ",round(Test_accuracy, 2))

print('-----------------------SVM ----------------------------------------')
dataset =Preprocessing_Our_Data("Regression",pd.read_csv('CarPrice_training.csv'),'Drop All Null Rows',False,False,True,True)
X_train_D, X_test_D, y_train_D, y_test_D = Split_Data(dataset,0.2,1,0,True,False)
r2score,train_time,model,Mean_Error,Train_accuracy,Test_accuracy= OUR_SVR(X_train_D, X_test_D, y_train_D, y_test_D)
print("Train Time = ",train_time)
print("R2_Score = ",round(r2score, 2))
print("Mse = ",round(Mean_Error, 2))
print("Train Acc = ",round(Train_accuracy, 2))
print("Test Acc = ",round(Test_accuracy, 2))
'''
print('-----------------------Decision Tree ----------------------------------------')
dataset =Preprocessing_Our_Data("Regression",pd.read_csv('CarPrice_training.csv'),'Drop All Null Rows',False,False,True,True)
X_train_D, X_test_D, y_train_D, y_test_D = Split_Data(dataset,0.2,1,0,True,False)
r2score,train_time,model,Mean_Error,Train_accuracy,Test_accuracy= OUR_Decision_Tree(X_train_D, X_test_D, y_train_D, y_test_D)
print("Train Time = ",train_time)
print("R2_Score = ",round(r2score, 2))
print("Mse = ",round(Mean_Error, 2))
print("Train Acc = ",round(Train_accuracy, 2))
print("Test Acc = ",round(Test_accuracy, 2))
'''

print('-----------------------LGBM ----------------------------------------')
dataset =Preprocessing_Our_Data("Regression",pd.read_csv('CarPrice_training.csv'),'Drop All Null Rows',False,False,True,True)
X_train_D, X_test_D, y_train_D, y_test_D = Split_Data(dataset,0.2,1,0,False,True)
r2score,train_time,model,Mean_Error,Train_accuracy,Test_accuracy= OUR_LGPM(X_train_D, X_test_D, y_train_D, y_test_D)
print("Train Time = ",train_time)
print("R2_Score = ",round(r2score, 2))
print("Mse = ",round(Mean_Error, 2))
print("Train Acc = ",round(Train_accuracy, 2))
print("Test Acc = ",round(Test_accuracy, 2))

print('-----------------------Random Forest ----------------------------------------')
dataset =Preprocessing_Our_Data("Regression",pd.read_csv('CarPrice_training.csv'),'Drop All Null Rows',False,False,True,True)
X_train_D, X_test_D, y_train_D, y_test_D = Split_Data(dataset,0.2,1,0,False,True)
r2score,train_time,model,Mean_Error,Train_accuracy,Test_accuracy= OUR_Random_Forest(X_train_D, X_test_D, y_train_D, y_test_D)
print("Train Time = ",train_time)
print("R2_Score = ",round(r2score, 2))
print("Mse = ",round(Mean_Error, 2))
print("Train Acc = ",round(Train_accuracy, 2))
print("Test Acc = ",round(Test_accuracy, 2))





def Plot():
    cars_dataset = pd.read_csv('CarPrice_training.csv')
    sns.pairplot(cars_dataset)
    plt.show()

def store():
    pkl_path = "./Car.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

def Load():
    pkl_path = './Car.pkl'
    with open(pkl_path, 'rb') as f:
        model = pickle.load(f)
    return model


