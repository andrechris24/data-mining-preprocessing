import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
print('IMPORT DATASET')
dataset=pd.read_csv('Dataset_Pembeli.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
print('Data X dengan missing value:')
print(x)
print('Data Y')
print(y)
print('\nMENGHILANGKAN MISSING VALUE')
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3]=imputer.transform(x[:, 1:3])
print('Data X tanpa missing value:')
print(x)
print('\nENCODING DATA KATEGORI (ATRIBUT)')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x=np.array(ct.fit_transform(x))
print('Data X dalam matrix:')
print(x)
print('\nENCODING DATA KATEGORI (CLASS/LABEL)')
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
print('Data Y dalam matrix:')
print(y)
print('\nPEMBAGIAN DATASET')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=1)
print('Dataset dalam training set')
print('X:')
print(x_train)
print('Y:')
print(y_train)
print('Dataset dalam test set')
print('X:')
print(x_test)
print('Y:')
print(y_test)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train[:, 3:]=sc.fit_transform(x_train[:, 3:])
x_test[:, 3:]=sc.transform(x_test[:, 3:])
print('\nData X setelah Feature Scaling')
print('Training set')
print(x_train)
print('Test Set')
print(x_test)
