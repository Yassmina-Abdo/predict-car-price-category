import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import *
from sklearn.svm import SVC
import time
from sklearn import tree
from tornado.autoreload import watch
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

Accuracy_Train = []
Accuracy_Test = []
Training_time = []
Testing_Time = []
model_name=[]

def Drawing_Graph():

    import matplotlib.pyplot as plt
    plt.bar(model_name, Accuracy_Train)
    plt.title(' All Train Accurecy')
    plt.xlabel('model')
    plt.ylabel('Train Accuracy')
    plt.show()

    import matplotlib.pyplot as plt
    plt.bar( model_name,Accuracy_Test)
    plt.title(' All Test Accurecy')
    plt.xlabel('model')
    plt.ylabel('Test Accuracy')
    plt.show()

    import matplotlib.pyplot as plt
    plt.bar( model_name,Training_time)
    plt.title('All Train Time ')
    plt.xlabel('model')
    plt.ylabel('Training Time')
    plt.show()

    import matplotlib.pyplot as plt
    plt.bar( model_name,Testing_Time)
    plt.title('All Test Time')
    plt.xlabel('model')
    plt.ylabel('Testing Time')
    plt.show()


def SelectBest_K (X_train, X_test, y_train, y_test):
    K_Range = range(1, 41)
    ERRors_List = []
    for i in K_Range:
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        ERRors_List.append(np.mean(y_pred != y_test))
    '''
    plt.plot(K_Range, ERRors_List, marker='o', markersize=7, markerfacecolor='blue', linestyle='dashed')
    plt.title('Error Rate K Value')
    plt.xlabel('The Value of K')
    plt.ylabel('Mean Error')
    plt.show()
    '''
    return 2

def SelectBest_Maxdepth (X_train, X_test, y_train, y_test):

    max_depth = range(1,51)
    ERRors_List = []
    for i in max_depth:
        model = tree.DecisionTreeClassifier(max_depth=i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        ERRors_List.append(np.mean(y_pred != y_test))
    '''
    plt.plot(max_depth, ERRors_List, marker='o', markersize=9, markerfacecolor='blue')
    plt.title('Error Rate max_depth Value')
    plt.xlabel('The Value of max_depth')
    plt.ylabel('Mean Error')
    #plt.show()
    '''
    # so the best Max_depth = 2

    return 2

def Make_BestScore(modelName,modelSearch,X_train, X_test, y_train, y_test):

    train_t = time.time()
    modelSearch.fit(X_train, y_train)
    if watch:
        train_time = round((time.time() - train_t), 4)
     # save the model to disk
    with open(modelName, 'wb') as file:
        pickle.dump(modelSearch, file)
    test_t = time.time()
    y_pred = modelSearch.predict(X_test)
    if watch:
        test_time = round((time.time() - test_t), 4)

    Mean_Error = metrics.mean_squared_error(y_test, y_pred)
    Train_accuracy = modelSearch.best_estimator_.score(X_train, y_train) * 100
    Test_accuracy = modelSearch.best_estimator_.score(X_test, y_test) * 100
    Accuracy_Train.append(Train_accuracy)
    Accuracy_Test.append(Test_accuracy)
    Training_time.append(test_time)
    Testing_Time.append(test_time)

    '''
    print("Best Accuracy : %f using %s" % (modelSearch.best_score_, modelSearch.best_params_))
    means = modelSearch.cv_results_['mean_test_score']
    for mean, param in zip(means, modelSearch.cv_results_['params']):
        print("%f  with: %r" % (mean, param))
    
    '''

    return train_time,test_time, Mean_Error, Train_accuracy, Test_accuracy

def Make_prediction(modelName,model,X_train, X_test, y_train, y_test):
    train_t = time.time()
    model.fit(X_train, y_train)
    if watch:
        train_time = round((time.time() - train_t), 4)
     # save the model to disk
    with open(modelName, 'wb') as file:
        pickle.dump(model, file)

    test_t = time.time()
    y_pred=model.predict(X_test)
    if watch:
        test_time = round((time.time() - test_t), 4)

    Mean_Error = metrics.mean_squared_error(y_test, model.predict(X_test))
    Train_accuracy = model.score(X_train,y_train) * 100
    Test_accuracy = metrics.accuracy_score(y_test,y_pred) * 100
    Accuracy_Train.append(Train_accuracy)
    Accuracy_Test.append(Test_accuracy)
    Training_time.append(test_time)
    Testing_Time.append(test_time)
    return train_time,test_time, Mean_Error, Train_accuracy, Test_accuracy

#-------------------------------------------------------------------------------------------
def OUR_KNN(X_train, X_test, y_train, y_test ):

    model = KNeighborsClassifier(n_neighbors= SelectBest_K (X_train, X_test, y_train, y_test))
    model_name.append('Knn')
    return Make_prediction('Knn_Model.pkl',model,X_train, X_test, y_train, y_test)


def OUR_LogisticRegression(X_train, X_test, y_train, y_test ):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    penalty = ['l1','l2']

    model = LogisticRegression(solver='liblinear')
    tuned_parameters = dict(C=C,penalty=penalty)
    modelSearch = GridSearchCV(model, tuned_parameters, scoring='accuracy', cv=4)

    model_name.append('Logistic Reg')
    return Make_BestScore('LogisticRegression_Model.pkl',modelSearch, X_train, X_test, y_train, y_test)

def OUR_DecisionTree_Classifier(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(max_depth=SelectBest_Maxdepth(X_train, X_test, y_train, y_test))
    model_name.append('Dec Tree')
    return Make_prediction('DecisionTreeClassifier.pkl',model, X_train, X_test, y_train, y_test)

def OUR_Adaboost_BY_DT(X_train, X_test, y_train, y_test):

    model = AdaBoostClassifier(n_estimators=100, learning_rate=0.01 ,algorithm='SAMME')
    model_name.append('Ada_Dt')
    return Make_prediction('Adaboost_ByDT_Model.pkl',model,X_train, X_test, y_train, y_test)


def OUR_Adaboost_BY_SVM(X_train, X_test, y_train, y_test ):

    svc = SVC(probability=True, kernel='linear')
    model = AdaBoostClassifier(n_estimators=100, base_estimator=svc,algorithm="SAMME")
    model_name.append('Ada_svm')
    return Make_prediction('Adaboost_BySVM_Model.pkl',model, X_train, X_test, y_train, y_test)




