# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression. 
2.Set variables for assigning dataset values. 
3.Import linear regression from sklearn. 
4.Predict the values of array. 
5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn. 
6.Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: NAMITHA.S
RegisterNumber: 212221040110 
*/
```
```
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
print("Placement data")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
print("Salary data")
data1.head()

print("Checking the null() function")
data1.isnull().sum()

print("Data Duplicate")
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
print("Print data")
data1

x=data1.iloc[:,:-1]
print("Data-status")
x

y=data1["status"]
print("data-status")
y


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("y_prediction array")
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy value")
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print("Confusion array")
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("Classification report")
print(classification_report1)

print("Prediction of LR")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![image](https://github.com/NamithaS2710/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133190822/c96f1e08-891b-4480-b0a8-796d469ff323)
![image](https://github.com/NamithaS2710/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133190822/1e033cbb-87f7-4a41-8284-d4193e52cfd5)
![image](https://github.com/NamithaS2710/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133190822/a5e4e0b1-454b-4647-a6d2-f90ec04920c2)
![image](https://github.com/NamithaS2710/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133190822/58d33a19-4324-4e8e-a35f-722d1c96368b)
![image](https://github.com/NamithaS2710/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133190822/107c58ec-baee-42b6-a35f-3955ff77758a)
![image](https://github.com/NamithaS2710/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133190822/936c705f-9238-46bb-9a29-99f8a2eefb86)
![image](https://github.com/NamithaS2710/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133190822/85b0e21d-eeaa-4ef6-8543-ccb43ec7b398)

![image](https://github.com/NamithaS2710/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133190822/b0dabd5c-ad23-4e86-9977-d54d04c0b727)
![image](https://github.com/NamithaS2710/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133190822/4dd08a93-8e44-4fdd-b26c-977eaa157b93)
![image](https://github.com/NamithaS2710/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/133190822/bbd2c8a8-432d-4022-a988-9d83e958d8d2)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
