import pandas as pd
import numpy as np
from numpy.linalg import inv
class MLR:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X_train, Y_train):
        try:
            if len(X_train) != len(Y_train):
                raise ValueError("The length of target variable and predictors are not the same.")
            else:
                X = np.hstack((np.ones((len(X_train),1)), X_train))
                betas = np.dot(inv(np.dot(X.T, X)), np.dot(X.T,Y_train)).round(3)
                np.set_printoptions(suppress=True, precision=3)
                self.intercept_ = betas[0]
                self.coef_ = betas[1:]
                print(self.intercept_, self.coef_)
        except ValueError as e:
            print(f"Error: {e}")    

    def predict(self, X_test):
        y_pred = np.dot(X_test, self.coef_) + self.intercept_
        return y_pred           

df = pd.read_csv('Data/Advertising.csv', index_col=0)
obj1 = MLR()
obj1.fit(df.iloc[: , :-1].values,  df.iloc[:, -1])
predicted  = obj1.predict([[23, 5, 6], [1, 5, 6]])
print(predicted)