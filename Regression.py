from sklearn.linear_model import LinearRegression,Ridge,Lasso,LassoCV
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from operator import itemgetter
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_squared_log_error,make_scorer
from sklearn.pipeline import make_pipeline
import time
from sklearn import tree
from tornado.autoreload import watch
import pickle

def OUR_Linear(X_train, X_test, y_train, y_test):
    regressor = LinearRegression()
    t=time.time()
    regressor.fit(X_train, y_train)
    if watch:
        myT=round((time.time()-t),4)
    # save the model to disk
    with open('Linear_Model.pkl', 'wb') as file:
        pickle.dump(regressor, file)
    y_pred = regressor.predict(X_test)
    R2_score= r2_score(y_test,y_pred)
    '''plt.style.use('ggplot')
    plt.scatter(y_test, y_pred, c='blue')
    plt.xlabel("Expected")
    plt.ylabel("Predicted value")
    plt.title("True value vs predicted value : Linear Regression")
    plt.show()'''

    return R2_score, myT,regressor,(metrics.mean_squared_error(y_test, y_pred)),(regressor.score(X_train, y_train) * 100),(regressor.score(X_test, y_test) * 100)

def OUR_Ridge(X_train, X_test, y_train, y_test):
    ridge = Ridge()
    parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 30, 40, 45, 50, 55, 60, 100]}
    ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=10)
    t = time.time()

    ridge_regressor.fit(X_train, y_train)
    if watch:
        myT = round((time.time() - t), 4)

    # save the model to disk
    with open('Ridge_Model.pkl', 'wb') as file:
        pickle.dump(ridge_regressor, file)
    y_pred_ridge = ridge_regressor.predict(X_test)
    R2_score = r2_score(y_test, y_pred_ridge)
    return R2_score, myT,ridge_regressor,(metrics.mean_squared_error(y_test, y_pred_ridge)),(ridge_regressor.best_estimator_.score(X_train, y_train)*100),(ridge_regressor.best_estimator_.score(X_test, y_test)*100)

def OUR_Lasso(X_train, X_test, y_train, y_test):
    lasso = Lasso()
    parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 30, 40, 45, 50, 55, 60, 100]}
    lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=10,)
    t = time.time()
    lasso_regressor.fit(X_train, y_train)
    if watch:
        myT = round((time.time() - t), 4)
    # save the model to disk
    with open('Lasso_Model.pkl', 'wb') as file:
        pickle.dump(lasso_regressor, file)
    y_pred_lasso = lasso_regressor.predict(X_test)
    R2_score = r2_score(y_test, y_pred_lasso)
    return R2_score,myT,lasso_regressor,(metrics.mean_squared_error(y_test, y_pred_lasso)), (
                lasso_regressor.best_estimator_.score(X_train, y_train) * 100), (
                       lasso_regressor.best_estimator_.score(X_test, y_test) * 100)

def OUR_SVR(X_train, X_test, y_train, y_test):
    SVR_module = SVR(kernel='linear')
    t = time.time()
    SVR_module.fit(X_train, y_train)
    if watch:
        myT = round((time.time() - t), 4)
    # save the model to disk
    with open('Svm_Model.pkl', 'wb') as file:
            pickle.dump(SVR_module, file)

    y_pred = SVR_module.predict(X_test)
    R2_score = r2_score(y_test, y_pred)
    return R2_score, myT,SVR_module,(metrics.mean_squared_error(y_test, y_pred)), (
            SVR_module.score(X_train, y_train) * 100), (
                   SVR_module.score(X_test, y_test) * 100)


def OUR_Decision_Tree(X_train, X_test, y_train, y_test):
    Trees = []
    y_train.reshape(len(y_train), 1)
    t = time.time()
    for i in range(100):
        Trees.append(DecisionTreeRegressor(random_state=i))
        Trees[i].fit(X_train, y_train)
    if watch:
        myT = round((time.time() - t), 4)
    y_pred = []
    scores = []
    y_test.reshape(len(y_test), 1)
    for i in range(100):
        y_pred.append(Trees[i].predict(X_test))
        scores.append((i, Trees[i].score(X_test, y_test)))

    best_randstate = max(scores, key=itemgetter(1))
    return myT,Trees,(best_randstate)



def OUR_Random_Forest(X_train, X_test, y_train, y_test):
    mod = RandomForestRegressor(n_estimators=100)
    model = make_pipeline(mod)
    t = time.time()
    model.fit(X_train, y_train)
    if watch:
        myT = round((time.time() - t), 4)
    # save the model to disk
    with open('RandomForestReg_Model.pkl', 'wb') as file:
        pickle.dump(model, file)

    y_pred = model.predict(X_test)
    R2_score = r2_score(y_test, y_pred)
    return R2_score, myT, model, (metrics.mean_squared_error(y_test, y_pred)), (
            model.score(X_train, y_train) * 100), (
                   model.score(X_test, y_test) * 100)


def OUR_LGPM(X_train, X_test, y_train, y_test):
    mod = LGBMRegressor(n_estimators=40)
    model = make_pipeline(mod)
    t = time.time()
    model.fit(X_train, y_train)
    if watch:
        myT = round((time.time() - t), 4)
    # save the model to disk
    with open('LGpm_Model.pkl', 'wb') as file:
            pickle.dump(model, file)

    y_pred = model.predict(X_test)
    R2_score = r2_score(y_test, y_pred)
    return R2_score, myT, model, (metrics.mean_squared_error(y_test, y_pred)), (
            model.score(X_train, y_train) * 100), (
                   model.score(X_test, y_test) * 100)






