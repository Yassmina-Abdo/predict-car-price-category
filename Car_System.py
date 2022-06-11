import Data_Handling,Regression
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
########################################## GUI INtERFACE ##############################################################################
Master=Tk()
Master.geometry("900x300")

#------------------------------------  Fill Menue ---------------------------------------------------------------------
Missing=StringVar()
OptionMenu(Master,Missing,'Drop All Null Rows','Fill With Forward Value','Fill With Backword Value','Mean','Median','Interpolate').place(x=230,y=40)
L1=Label(Master,text="Missing Value")
L1.place(x=100,y=40)
L1.configure(font='helvetica 12 bold ')
#------------------------------------  Fill Menue ---------------------------------------------------------------------


#------------------------------------ All_data Box ----------------------------------------------------------------------
all_data=BooleanVar()
Checkbutton(Master,text="All_data",font='helvetica 12 bold ',variable=all_data).place(x=500, y=40)
#------------------------------------ Bias Check Box ----------------------------------------------------------------------

#------------------------------------ Rank Box ----------------------------------------------------------------------
Rank=BooleanVar()
Checkbutton(Master,text="Rank",font='helvetica 12 bold ',variable=Rank).place(x=600, y=40)
#------------------------------------ Rank Box ----------------------------------------------------------------------

#------------------------------------ Scaler Box ----------------------------------------------------------------------
Scaler=BooleanVar()
Checkbutton(Master,text="Scaler",font='helvetica 12 bold ',variable=Scaler).place(x=700, y=40)
#------------------------------------ Scaler Box ----------------------------------------------------------------------



############################################################################################################################################################


#------------------------------------ L test_size TextBox ----------------------------------------------------------------------
test_size=Entry(Master)
test_size.place(x=230,y=115)
test_size.focus_set()
L3=Label(Master,text="Test_Size")
L3.place(x=100,y=115)
L3.configure(font='helvetica 12 bold ')
#------------------------------------ L test_size TextBox ----------------------------------------------------------------------



#------------------------------------ random_state Box ----------------------------------------------------------------------
random_state=IntVar()
Checkbutton(Master,text="Random_state",font='helvetica 12 bold ',variable=random_state).place(x=500, y=115)
#------------------------------------ random_state Box ----------------------------------------------------------------------


#------------------------------------ shuffle Box ----------------------------------------------------------------------
shuffle=IntVar()
Checkbutton(Master,text="Shuffle",font='helvetica 12 bold ',variable=shuffle).place(x=700, y=115)
#------------------------------------ shuffle Box ----------------------------------------------------------------------


#########################################################################################################################################

#------------------------------------  Regression Menue ---------------------------------------------------------------------
regression=StringVar()
OptionMenu(Master,regression,'Linear','Ridge','Lasso','SVR','Decision_Tree','LGBM','Random_Forest').place(x=230,y=170)
L1=Label(Master,text="Regression")
L1.place(x=100,y=170)
L1.configure(font='helvetica 12 bold ')
#------------------------------------  Regression Menue ---------------------------------------------------------------------


model=None
#------------------------------------ EVALUATE BUTTON -----------------------------------------------------------------------


def Evaluate_button(model1):
    if (regression.get() == 'Decision_Tree'):
        Data_D = Data_Handling.Preprocessing_Our_Data("Regression",pd.read_csv('CarPrice_training.csv'),Missing.get(), all_data.get(), Rank.get(), Scaler.get(), True)
        X_train_D, X_test_D, y_train_D, y_test_D = Data_Handling.Split_Data(Data_D, float(test_size.get()),
                                                                                    random_state.get(), shuffle.get(),
                                                                                    True,False)
        myT, model, bestACC_D = Regression.OUR_Decision_Tree(X_train_D, X_test_D, y_train_D, y_test_D)
        Label(Master, text="Mean Squad =                                                          ",font='helvetica 12 bold ').place(x=600, y=200)
        Label(Master, text="Train =                                                              %",font='helvetica 12 bold ').place(x=600, y=230)
        Label(Master, text="Test  =                        %", font='helvetica 12 bold ').place(x=600, y=260)
        Label(Master, text="Time  =                         ", font='helvetica 12 bold ').place(x=600, y=280)
        Label(Master, text=round((bestACC_D[1] * 100), 2), font='helvetica 12 bold ', bg='yellow').place(x=690,y=260)
        Label(Master, text=myT, font='helvetica 12 bold ', bg='yellow').place(x=690, y=280)

    elif (regression.get() == 'Random_Forest'):
        Data_RF = Data_Handling.Preprocessing_Our_Data("Regression",pd.read_csv('CarPrice_training.csv'),Missing.get(), all_data.get(), Rank.get(), Scaler.get(), True)
        X_train_RF, X_test_RF, y_train_RF, y_test_RF = Data_Handling.Split_Data(Data_RF, float(test_size.get()),
                                                                                    random_state.get(), shuffle.get(),False,
                                                                                    True)
        myT,model,bestACC_RF = Regression.OUR_Random_Forest(X_train_RF, X_test_RF, y_train_RF, y_test_RF)
        Label(Master, text="Mean Squad =                                                             ", font='helvetica 12 bold ').place(x=600, y=200)
        Label(Master, text="Train =                                                                 %", font='helvetica 12 bold ').place(x=600, y=230)
        Label(Master, text="Test  =                        %", font='helvetica 12 bold ').place(x=600, y=260)
        Label(Master, text="Time  =                         ", font='helvetica 12 bold ').place(x=600, y=280)
        Label(Master, text=round((bestACC_RF*100),2), font='helvetica 12 bold ', bg='yellow').place(x=690, y=260)
        Label(Master, text=myT, font='helvetica 12 bold ', bg='yellow').place(x=690, y=280)
    elif (regression.get() == 'LGBM'):
        Data_LGBM = Data_Handling.Preprocessing_Our_Data("Regression",pd.read_csv('CarPrice_training.csv'),Missing.get(), all_data.get(), Rank.get(), Scaler.get(), True)
        X_train_LGBM, X_test_LGBM, y_train_LGBM, y_test_LGBM = Data_Handling.Split_Data(Data_LGBM, float(test_size.get()),
                                                                                    random_state.get(), shuffle.get(),False,
                                                                                    True)
        myT,model,bestACC_LGBM = Regression.OUR_LGPM(X_train_LGBM, X_test_LGBM, y_train_LGBM, y_test_LGBM)
        Label(Master, text="Mean Squad =                                                             ", font='helvetica 12 bold ').place(x=600, y=200)
        Label(Master, text="Train =                                                                 %", font='helvetica 12 bold ').place(x=600, y=230)
        Label(Master, text="Test  =                        %", font='helvetica 12 bold ').place(x=600, y=260)
        Label(Master, text="Time  =                         ", font='helvetica 12 bold ').place(x=600, y=280)
        Label(Master, text=round((bestACC_LGBM*100),2), font='helvetica 12 bold ', bg='yellow').place(x=690, y=260)
        Label(Master, text=myT, font='helvetica 12 bold ', bg='yellow').place(x=690, y=280)
    else:
        Data = Data_Handling.Preprocessing_Our_Data("Regression",pd.read_csv('CarPrice_training.csv'),Missing.get(), all_data.get(), Rank.get(), Scaler.get(), False)
        X_train, X_test, y_train, y_test = Data_Handling.Split_Data(Data, float(test_size.get()), random_state.get(),
                                                                    shuffle.get(), False, False)
        if (regression.get() == 'Linear'):
            myT,model,Error, Train, Test = Regression.OUR_Linear(X_train, X_test, y_train, y_test)
        elif (regression.get() == 'Ridge'):
            myT,model,Error, Train, Test = Regression.OUR_Ridge(X_train, X_test, y_train, y_test)
        elif (regression.get() == 'Lasso'):
            myT,model,Error, Train, Test = Regression.OUR_Lasso(X_train, X_test, y_train, y_test)
        else:
            Data_SVR = Data_Handling.Preprocessing_Our_Data("Regression",pd.read_csv('CarPrice_training.csv'),Missing.get(), all_data.get(), Rank.get(), Scaler.get(),True)
            X_train_SVR, X_test_SVR, y_train_SVR, y_test_SVR = Data_Handling.Split_Data(Data_SVR, float(test_size.get()),
                                                                        random_state.get(), shuffle.get(),True,False)
            myT,model,Error, Train, Test = Regression.OUR_SVR(X_train_SVR, X_test_SVR, y_train_SVR, y_test_SVR)

        Label(Master, text="Mean Squad =                        ", font='helvetica 12 bold ').place(x=600, y=200)
        Label(Master, text="Train =                        %", font='helvetica 12 bold ').place(x=600, y=230)
        Label(Master, text="Test  =                        %", font='helvetica 12 bold ').place(x=600, y=260)
        Label(Master, text="Time  =                         ", font='helvetica 12 bold ').place(x=600, y=280)
        Label(Master, text=round(Error, 2), font='helvetica 12 bold ', bg='yellow').place(x=715, y=200)
        Label(Master, text=round(Train, 2), font='helvetica 12 bold ', bg='yellow').place(x=690, y=230)
        Label(Master, text=round(Test, 2), font='helvetica 12 bold ', bg='yellow').place(x=690, y=260)
        Label(Master, text=myT, font='helvetica 12 bold ', bg='yellow').place(x=690, y=280)
Button(Master, text="Evaluate", font='helvetica 8 bold ', bg='blue', height=3, width=30,command=lambda: Evaluate_button(model)).place(x=330, y=220)
#------------------------------------ EVALUATE BUTTON -----------------------------------------------------------------------

def Plot():
    cars_dataset = pd.read_csv('CarPrice_training.csv')
    sns.pairplot(cars_dataset)
    plt.show()
Button(Master, text="Plot All DATA", font='helvetica 8 bold ', bg='blue', height=1, width=30,command=lambda: Plot()).place(x=10, y=220)

def store():
    pkl_path = "./Car.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
Button(Master, text="Store Model", font='helvetica 8 bold ', bg='blue', height=1, width=30,command=lambda: store()).place(x=10, y=245)

def Load():
    pkl_path = './Car.pkl'
    with open(pkl_path, 'rb') as f:
        model = pickle.load(f)
    return model
Button(Master, text="Load Model", font='helvetica 8 bold ', bg='blue', height=1, width=30,command=lambda: Load()).place(x=10, y=270)

##########################################  GUI INtERFACE #######################################################################################
Master.mainloop()


