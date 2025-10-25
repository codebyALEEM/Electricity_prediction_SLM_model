import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle 

dataset = pd.read_csv(r'C:\Users\VICTUS\Desktop\mastering git\Practise git\Electricity_prediction_SLM_model\electricity_bill.csv')

X = dataset.iloc[:,0:1]
y = dataset.iloc[:,1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)

filename = 'SLR_Electricity_Prediction.pkl'
with open(filename,'wb') as file:
    pickle.dump(regressor,file)
print("Model has been pickled and saved as SLR_Electricity_Prediction.pkl") 