# EX-04 Implementation of Logistic Regression Model to Predict the Placement Status of Student
### Aim:
To write a program to implement the the Logistic Regression Model to Predict the &emsp;&emsp;&emsp;&emsp;&emsp;**DATE:** <br>Placement Status of Student.
### Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
### Algorithm
1. Get the data and use label encoder to change all the values to numeric.
2. Drop the unwanted values,Check for NULL values, Duplicate values. 
3. Classify the training data and the test data. 
4. Calculate the accuracy score, confusion matrix and classification report.
### Program:
```
Developed By: YUVASAKTHI N.C
Register No: 212222240120
```
```Python
import pandas as pd
df=pd.read_csv('CSVs/Placement_Data.csv')
df.head()
df=df.drop(['sl_no','salary'],axis=1)
df.isnull().sum()
df.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
l=["gender","ssc_b","hsc_b","hsc_s","degree_t","workex","specialisation","status"]
for i in l:
    df[i]=le.fit_transform(df[i])
df.head()
x=df.iloc[:,:-1]
x.head()
y=df["status"]
y.head()
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear") 
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("Confusion matrix:\n",confusion)
from sklearn.metrics import classification_report
ClsfctonReprt=classification_report(y_test,y_pred)
print(ClsfctonReprt)
```
### Output:
### Head of the data:
#### Values:

![image](https://github.com/IamShakthi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117913445/f0e71c61-cba7-4067-87e9-97a7a2c6f4e9)

#### Null:

![image](https://github.com/IamShakthi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117913445/fb86d529-541a-4dba-8a6e-8c7bbe9ebc5a)

#### Transformed Data:
![image](https://github.com/IamShakthi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117913445/a62190b4-17b2-4398-a589-e1fda2509bf6)


#### X values:
![image](https://github.com/IamShakthi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117913445/f135226b-1c91-4a5e-8093-8f83b6af32a9)


#### Y values :
![image](https://github.com/IamShakthi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117913445/352243ae-7615-418a-acd4-eaeeaea2ed62)

#### Y Predicted :
![image](https://github.com/IamShakthi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117913445/84c59ec8-6c3a-4979-90f1-55e16de5d7e8)

#### Accuracy :
![image](https://github.com/IamShakthi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117913445/8a7b4202-227e-447b-a24c-80953ea64ce7)


#### Confusion Matrix :
![image](https://github.com/IamShakthi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117913445/e5cdeb8f-6dd4-4804-91dc-adb74a8986e6)


#### Classification Report :

![image](https://github.com/IamShakthi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117913445/b71748a1-b228-49fe-9fec-6cd2588c92f2)



### Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
