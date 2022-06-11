import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler,LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split


def Preprocessing_Our_Data(Type,cars_dataset,fill,all_data,Rank,Scaler,SVR_Decission_LGBM_RandomForest):

   cars_dataset=cars_dataset.copy()
   if(Type=='Regression'):
       cars_dataset = FillOptions(cars_dataset, fill)
       # -------------------------------------- check for unique of car name ---------------------------------
       cars_dataset['CarName'] = cars_dataset['CarName'].str.split(' ', expand=True)
       Cars1 = [cars_dataset['CarName']]
       # print(Cars1)
       Cars2 = [cars_dataset['CarName'].unique()]
       # print(Cars2)
       cars_dataset['CarName'] = cars_dataset['CarName'].replace(
           {'maxda': 'mazda', 'nissan': 'Nissan', 'porcshce': 'porsche', 'toyouta': 'toyota'})
       Cars3 = [cars_dataset['CarName'].unique()]
       # print(Cars3)
       # -------------------------------------- check for unique of car name ---------------------------------
   elif('Classification'):
       cars_dataset = cars_dataset.fillna({"category": "High"})
       cars_dataset= FillOptions(cars_dataset, fill)
   # ----------------------------------- check about duplicate of columns -------------------------------
   '''print("--------------------------Duplicated--------------------------")
   D_Columns = cars_dataset.loc[cars_dataset.duplicated()]
   print(D_Columns)
   print("----------------------------------------------------")'''
   # ----------------------------------- check about duplicate of columns -------------------------------
   if(SVR_Decission_LGBM_RandomForest==False):
       D = Encoding(cars_dataset,Type)
       cars_dataset = Data_Analysis(Type,D, all_data=all_data, Rank=Rank, Scaler=Scaler)

   return cars_dataset


def Encoding(cars_dataset,Type):

    #  By Label Encoding
    lb = LabelEncoder()
    cars_dataset['enginelocation'] = lb.fit_transform(cars_dataset['enginelocation'])
    cars_dataset['cylindernumber'] = lb.fit_transform(cars_dataset['cylindernumber'])
    cars_dataset["fuelsystem"] = lb.fit_transform(cars_dataset["fuelsystem"])
    #  By One Hot
    cars_dataset = pd.get_dummies(cars_dataset,
                                  columns=["CarName", "fueltype", "aspiration", "doornumber", "carbody", "drivewheel",
                                           "enginetype"],
                                  prefix=["CarName", "fueltype", "aspiration", "doornumber", "carbody", "drivewheel",
                                          "enginetype"])

    if (Type == 'Regression'):
        temp_price = cars_dataset['price']
        cars_dataset.drop(columns=['price'], inplace=True)
        cars_dataset['price'] = temp_price
    elif ('Classification'):
        Replace_Category = {"category": {"High": 1, "Low": 0}}
        cars_dataset = cars_dataset.replace(Replace_Category)

        temp_price=cars_dataset['category']
        cars_dataset.drop(columns=['category'],inplace=True)
        cars_dataset['category']=temp_price


    return cars_dataset


def Data_Analysis(Type,cars_dataset, all_data, Rank, Scaler):
    '''sns.pairplot(cars_dataset)
    plt.show()'''
    if (all_data == True): # ---------Take Only All Numerical Data then check if doing Rank and Scaler or not---
        #print(cars_dataset.dtypes)
        listofheaders=[]
        Not_Object=cars_dataset.select_dtypes(include= ['int64'] and ['float64'])
        alllistofheader=Not_Object.columns
        for i in range(len(alllistofheader)-1):
            listofheaders.append(alllistofheader[i])

        OurData = Check_Rank_Nnd_Scaler(Rank, Scaler, listofheaders, cars_dataset)
    else:  # ---------------------------Take Data By best Correlation---------------
        data_after_corr,listofheaders=CorrelationData(cars_dataset, Type)
        OurData = Check_Rank_Nnd_Scaler(Rank,Scaler,listofheaders,data_after_corr)

    return OurData


def Split_Data(cars_dataset,test_size,random_state,shuffle,SVR_Decission,LGBM_RandomForest):

    if(SVR_Decission==True and LGBM_RandomForest==False):
        cars_dataset['carwidth'] = cars_dataset['carwidth'].rank()
        cars_dataset['wheelbase'] = cars_dataset['wheelbase'].rank()
        cars_dataset.drop(columns=['CarName', 'doornumber', 'carheight', 'stroke', 'compressionratio', 'peakrpm'],
                          inplace=True)

        X = cars_dataset.iloc[:, :-1].values
        Y = cars_dataset.iloc[:, -1].values
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 2, 3, 4, 5, 10, 11, 13])],
                               remainder='passthrough')
        X = ct.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state,shuffle=shuffle)

        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        y_train = sc_y.fit_transform(y_train.reshape(len(y_train), 1))
        X_test = sc_X.transform(X_test)
        y_test = sc_y.transform(y_test.reshape(len(y_test), 1))
    elif(SVR_Decission==False and LGBM_RandomForest==True):

        lab = LabelEncoder()
        cars_dataset['fuelsystem'] = lab.fit_transform(cars_dataset['fuelsystem'])
        cars_dataset['cylindernumber'] = lab.fit_transform(cars_dataset['cylindernumber'])
        cars_dataset['enginetype'] = lab.fit_transform(cars_dataset['enginetype'])
        cars_dataset['enginelocation'] = lab.fit_transform(cars_dataset['enginelocation'])
        cars_dataset['drivewheel'] = lab.fit_transform(cars_dataset['drivewheel'])
        cars_dataset['carbody'] = lab.fit_transform(cars_dataset['carbody'])
        cars_dataset['doornumber'] = lab.fit_transform(cars_dataset['doornumber'])
        cars_dataset['aspiration'] = lab.fit_transform(cars_dataset['aspiration'])
        cars_dataset['fueltype'] = lab.fit_transform(cars_dataset['fueltype'])
        cars_dataset['CarName'] = lab.fit_transform(cars_dataset['CarName'])

        cars_dataset['enginesize'] = cars_dataset['enginesize'].rank()

        X = cars_dataset.drop(['price'], axis=1)
        Y = cars_dataset['price']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
        '''X_train=X
        y_train=Y'''
    else:
        X = cars_dataset.iloc[:, :-1].values
        Y = cars_dataset.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state,
                                                            shuffle=shuffle)

    return X_train, X_test, y_train, y_test



def CorrelationData(cars_dataset,Type):
    # ------------------------------------ see Correlation -------------------------------------
    corr = cars_dataset.corr()
    # plt.subplots(figsize=(15, 15))
    # sns.heatmap(corr, annot=True)
    # plt.show()
    if (Type == 'Regression'):
        top_feature = corr.index[abs(corr['price'] > 0.5)]
    elif ('Classification'):
        top_feature = corr.index[abs(corr['category'] > 0.5)]
    '''
    plt.subplots(figsize=(15, 15))
    top_corr = cars_dataset[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    #plt.show()
    '''
    # ------------------------------------ see Correlation -------------------------------------
    data_after_corr = pd.DataFrame(cars_dataset[top_feature])
    listofheaders = []
    Not_Object = data_after_corr.select_dtypes(include=['int64'] and ['float64'])
    alllistofheader = Not_Object.columns
    for i in range(len(alllistofheader) - 1):
        listofheaders.append(alllistofheader[i])

    return  data_after_corr,listofheaders



def Check_Rank_Nnd_Scaler(Rank,Scaler,listofheaders,data_after_corr):

    if (Rank == True):
        for f in listofheaders:
            # sns.boxplot(data=data_after_corr[f], palette='Pastel2')
            # plt.show()
            data_after_corr[f] = data_after_corr[f].rank()
            # sns.boxplot(data=data_after_corr[f], palette='Pastel2')
            # plt.show()
    if (Scaler == True):
        for f in listofheaders:
            sds = StandardScaler()
            data_after_corr[[f]] = sds.fit_transform(data_after_corr[[f]])
    return  data_after_corr



def FillCategories(cars_dataset):

    cars_dataset = cars_dataset.fillna({"fueltype": "gas"})
    cars_dataset = cars_dataset.fillna({"aspiration": "std"})
    cars_dataset = cars_dataset.fillna({"enginelocation": "front"})
    cars_dataset = cars_dataset.fillna({"enginetype": "ohc"})
    cars_dataset = cars_dataset.fillna({"cylindernumber": "four"})
    cars_dataset = cars_dataset.fillna({"fuelsystem": "2bbl"})
    # -------------------------------------- check for unique of car name ---------------------------------
    cars_dataset['CarName'] = cars_dataset['CarName'].str.split(' ', expand=True)
    Cars1 = [cars_dataset['CarName']]
    # print(Cars1)
    Cars2 = [cars_dataset['CarName'].unique()]
    # print(Cars2)
    cars_dataset['CarName'] = cars_dataset['CarName'].replace(
        {'maxda': 'mazda', 'nissan': 'Nissan', 'porcshce': 'porsche', 'toyouta': 'toyota'})
    Cars3 = [cars_dataset['CarName'].unique()]
    # print(Cars3)
    # -------------------------------------- check for unique of car name ---------------------------------

    return cars_dataset


def FillOptions(cars_dataset,fill):

    if (fill == 'Drop All Null Rows'):
        cars_dataset = cars_dataset.dropna(axis=0)

    elif (fill == 'Fill With Forward Value'):

        cars_dataset = FillCategories(cars_dataset)
        cars_dataset = cars_dataset.fillna(method='pad')
    elif (fill == 'Fill With Backword Value'):

        cars_dataset = FillCategories(cars_dataset)
        cars_dataset = cars_dataset.fillna(method='bfill')
    elif (fill == 'Mean'):

        cars_dataset = FillCategories(cars_dataset)
        cars_dataset = cars_dataset.fillna(cars_dataset.mean())
    elif (fill == 'Median'):

        cars_dataset = FillCategories(cars_dataset)
        cars_dataset = cars_dataset.fillna(cars_dataset.median())
    else:

        cars_dataset = FillCategories(cars_dataset)
        cars_dataset = cars_dataset.fillna(cars_dataset.interpolate())

    return cars_dataset







