
# coding: utf-8

# In[8]:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron:
  
    def __init__ (self):
        self.w = None
        self.b = None
    
    def model(self, x):
        return 1 if (np.dot(self.w, x) >= self.b) else 0
    
    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)
    
    def fit(self, X, Y, epochs = 1, lr = 1):
    
        self.w = np.ones(X.shape[1])
        self.b = 0

        accuracy = {}
        max_accuracy = 0

        wt_matrix = []
    
        for i in range(epochs):
            for x, y in zip(X, Y):
                y_pred = self.model(x)
                if y == 1 and y_pred == 0:
                    self.w = self.w + lr * x
                    self.b = self.b - lr * 1
                elif y == 0 and y_pred == 1:
                    self.w = self.w - lr * x
                    self.b = self.b + lr * 1

            wt_matrix.append(self.w)    

            accuracy[i] = accuracy_score(self.predict(X), Y)
            if (accuracy[i] > max_accuracy):
                max_accuracy = accuracy[i]
                chkptw = self.w
                chkptb = self.b

        self.w = chkptw
        self.b = chkptb

        print(max_accuracy)

        return np.array(wt_matrix)

traininp = pd.read_csv("C:\\Users\\mkanukolanu\\Downloads\\padhai-module1-assignment\\train.csv")
testdf = pd.read_csv("C:\\Users\\mkanukolanu\\Downloads\\padhai-module1-assignment\\test.csv")

traininp['class'] = traininp['Rating'].map(lambda x:1 if x>=4 else 0)

X_raw = traininp.drop(['PhoneId','Rating','class'],axis=1)
X = X_raw.notnull().astype('int')

Y = traininp[['class']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify = Y, random_state=1)

X_test = X_test.values
X_train = X_train.values

perceptron = Perceptron()
wt_matrix = perceptron.fit(X_train, Y_train, 10000, 0.0001)

Y_test_pred = perceptron.predict(X_test)
accuracy_test = accuracy_score(Y_test_pred, Y_test)
print(accuracy_test)

X1_raw = testdf.drop(['PhoneId'],axis=1)
X1 = X1_raw.notnull().astype('int')
X1 = X1.values

Y1_pred = perceptron.predict(X1)

finaldf = pd.concat([testdf[['PhoneId']], pd.DataFrame(Y1_pred.astype('int'))], axis=1)
finaldf.columns = ['PhoneId','Class']

finaldf.to_csv("C:\\Users\\mkanukolanu\\Downloads\\padhai-module1-assignment\\percsubmission1.csv",index=False)

