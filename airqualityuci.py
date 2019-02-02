# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 18:45:31 2018

@author: Narayanan Abishek
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LinearRegression  

#original dataset
def get_data():
    airquality_df = pd.read_csv('D:\Projects\Kaggle\Datasets\AirQuality UCI\AirQualityUCI.csv',sep=';',decimal=',')
    airquality_df = airquality_df.fillna(value = 0, axis = 1)  #filling null with zeros 
    airquality_df = airquality_df.drop(['Time'],axis = 1)

    return airquality_df
    
#feature scaling for the gradient descent to converge faster
def feature_scaling(x,y):
    
    X_mean = np.mean(x,axis=0)
    X_sigma = np.std(x,axis=0)
    Y_mean = np.mean(y,axis=0)
    Y_sigma = np.std(y,axis=0)
    X = np.divide((x - X_mean),X_sigma)
    Y = np.divide((y - Y_mean), Y_sigma)
    return X,Y
    
def model():   
    
    my_df = get_data()
    x = my_df.iloc[:,1:13]
    y = my_df.iloc[:,13]
    
    X = np.reshape(np.array(x),(x.shape[0],x.shape[1]))
    Y = np.reshape(np.array(y),(y.shape[0],1))   
        
    from sklearn.model_selection import train_test_split  
    
    train_x, val_x, train_y,val_y = train_test_split(X, Y, test_size=0.2, random_state=0) 
    
    train_x, test_x, train_y,test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=0) 
    
    train_X,train_Y = feature_scaling(train_x,train_y)
    
    val_X,val_Y = feature_scaling(val_x,val_y)
    test_X,test_Y = feature_scaling(test_x,test_y)
    regressor = LinearRegression()  
    regressor.fit(train_X, train_Y)
    
    val_Y_pred = regressor.predict(val_X)
    
    from sklearn import metrics 

    print("\nValidation set accuracy: \n")
    print('Mean Absolute Error:', metrics.mean_absolute_error(val_Y, val_Y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(val_Y, val_Y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(val_Y, val_Y_pred)))
    print('Explained Variance Score:', explained_variance_score(val_Y,val_Y_pred,multioutput='raw_values')*100,"%")
    
    print("\nTesting set accuracy: \n")
    
    test_Y_pred = regressor.predict(test_X)
    
    print('Mean Absolute Error:', metrics.mean_absolute_error(test_Y, test_Y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(test_Y, test_Y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_Y, test_Y_pred)))
    print('Explained Variance Score:', explained_variance_score(test_Y, test_Y_pred,multioutput='raw_values')*100,"%")
   
    #print("\n\n Actual values:\n",test_Y,"\n\n Predicted values:\n",test_Y_pred)
    print(regressor.coef_)
        
model()