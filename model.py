

# DECISION TREE MODEL

'''
This model predicts the classification of iris flower based on sepal length, 
sepal width, petal length and petal width using Decision tree model.

'''

#<!------------------------ IMPORTING LIBRARIES ---------------------->

import pandas as pd 
import matplotlib.pyplot as plt


#<!------------------------ LOAD THE DATA ---------------------->

df = pd.read_excel(r"C:\Users\athir\Iris_.xls")
df

#<!---TOTAL NUMBER OF NULL VALUES IN EACH FEATURES IN THE DATAFRAME.---> 
df.isna().sum()

#<!---STATISTICS SUMMARY OF THE GIVEN DATAFRAME. ---> 
df.describe()


#<!------------------------ VISUALIZATION ---------------------->

df.corr()

fig = plt.figure(figsize=(12,10))

ax1=plt.subplot(221)
ax1.scatter(df.SL,df.PL)
plt.xlabel('sepal_length')
plt.ylabel('petal_length')

ax2=plt.subplot(222)
ax2.scatter(df.PW,df.PL)
plt.xlabel('petal_width')
plt.ylabel('petal_length')


#<!------------------------ TRAIN,TEST SPLIT ---------------------->

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
X_train
X_test
y_train


#<!------------------------ MODELING ---------------------->


# Fitting Decision Tree Model to the Training set
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(max_depth=4)
DT.fit(X_train,y_train)
DT.score(X_train, y_train)
DT.score(X_test, y_test)

# Predicting the Test set results
y_test_pred = DT.predict(X_test)

X_test,y_test


#<!--------------- SAVE THE MODEL AS A PRECOMPILED PKL FILE -------------->
import pickle
pickle.dump(DT, open('model.pkl', 'wb'))  

#<!---------------  LOAD THE MODEL TO COMPARE THE RESULTS ---------------->

model = pickle.load(open('model.pkl','rb')) 
print(model.predict([[6.1,2.8,4.7,1.2]]))


