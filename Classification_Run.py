from Data_Handling  import *
from Regression import  *
from  Classification import  *

Accuracy_Train = []
Accuracy_Test = []
Training_time = []
Testing_Time = []
model_name=[]

print('----------------------KNN-----------------------------------------')
dataset = Preprocessing_Our_Data("Classification",pd.read_csv('CarPrice_training_classification.csv'),"Mean",False,True,True,False)
X_train, X_test, y_train, y_test = Split_Data(dataset,0.2,2,True,'','')
train_time,test_time, Mean_Error,Train_accuracy, Test_accuracy= OUR_KNN(X_train, X_test, y_train, y_test )
print("Train Time = ",train_time)
print("Test Time = ",test_time)
print("Mse = ",Mean_Error)
print("Train Acc = ",Train_accuracy) #94.48
print("Test Test = ",Test_accuracy)  #93.75
print('-----------------------LOGISTIC Reg----------------------------------------')

dataset = Preprocessing_Our_Data("Classification",pd.read_csv('CarPrice_training_classification.csv'),"Mean",False,False,True,False)
X_train, X_test, y_train, y_test = Split_Data(dataset,0.2,4,True,'','')
train_time,test_time,Mean_Error,Train_accuracy, Test_accuracy=OUR_LogisticRegression(X_train, X_test, y_train, y_test )
print("Train Time = ",train_time)
print("Test Time = ",test_time)
print("Mse = ",Mean_Error)
print("Train Acc = ",Train_accuracy)  # 93.70
print("Test Test = ",Test_accuracy)   # 93.75


print('---------------------Decision Tree------------------------------------------')

dataset = Preprocessing_Our_Data("Classification",pd.read_csv('CarPrice_training_classification.csv'),"Mean",False,False,True,False)
X_train, X_test, y_train, y_test = Split_Data(dataset,0.2,1,False,'','')
train_time,test_time, Mean_Error,Train_accuracy, Test_accuracy=OUR_DecisionTree_Classifier(X_train, X_test, y_train, y_test )
print("Train Time = ",train_time)
print("Test Time = ",test_time)
print("Mse = ",Mean_Error)
print("Train Acc = ",Train_accuracy)  # 95.27
print("Test Test = ",Test_accuracy)   # 93.75

print('--------------------Adaboost Dt-------------------------------------------')
#   For Adaboost BY DT
dataset = Preprocessing_Our_Data("Classification",pd.read_csv('CarPrice_training_classification.csv'),"Mean",False,False,True,False)
X_train, X_test, y_train, y_test = Split_Data(dataset,0.2,1,True,'','')
train_time,test_time, Mean_Error,Train_accuracy, Test_accuracy=OUR_Adaboost_BY_DT(X_train, X_test, y_train, y_test )
print("Train Time = ",train_time)
print("Test Time = ",test_time)
print("Mse = ",Mean_Error)
print("Train Acc = ",Train_accuracy)  # 92.91
print("Test Test = ",Test_accuracy)   # 93.75  96 Sometimes

print('---------------------Adaboost Svm------------------------------------------')
#   For Adaboost BY SVM
dataset = Preprocessing_Our_Data("Classification",pd.read_csv('CarPrice_training_classification.csv'),"Mean",False,False,True,False)
X_train, X_test, y_train, y_test = Split_Data(dataset,0.2,1,True,'','')
train_time,test_time, Mean_Error,Train_accuracy, Test_accuracy=OUR_Adaboost_BY_SVM(X_train, X_test, y_train, y_test )
print("Train Time = ",train_time)
print("Test Time = ",test_time)
print("Mse = ",Mean_Error)
print("Train Acc = ",Train_accuracy)  # 92.91
print("Test Test = ",Test_accuracy)   # 93.75



#Drawing_Graph()