import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.svm import SVC


data=pd.read_csv("Training.csv").dropna(axis = 1)
print(data)
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])
X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.2, random_state = 24)

svm_model = SVC()
s=svm_model.fit(X_train, y_train)
pickle.dump(s,open('model1.pkl','wb'))
