import pandas as pd;
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import pickle


df = pd.read_csv('iris.csv')
dummies = pd.get_dummies(df.Species)
# print(df)
df_dummies = pd.concat([df,dummies],axis='columns')
df_dummies = df_dummies.drop(['Species'],axis='columns')
X_train,X_test,Y_train,Y_test = train_test_split(df.drop(['Species','Id'],axis='columns'),df.Species)
model = LogisticRegression(solver='lbfgs', max_iter=150, multi_class='multinomial', penalty='l2')
model.fit(X_train,Y_train)
# result = model.predict([[1.0,2.0,3.0,4.0]])
# print(f"Hello, my name is {result[0]} and I am {result[0]} years old." )
# model.score(Y_train,Y_test)
pickle.dump(model,open('model.pkl','wb'))
