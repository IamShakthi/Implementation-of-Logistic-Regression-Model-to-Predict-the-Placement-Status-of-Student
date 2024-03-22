# EX-04 Implementation of Logistic Regression Model to Predict the Placement Status of Student
### Aim:
To write a program to implement the the Logistic Regression Model to Predict the &emsp;&emsp;&emsp;&emsp;&emsp;**DATE:** <br>Placement Status of Student.
### Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
### Algorithm
1. Get the data and use label encoder to change all the values to numeric.
2. Drop the unwanted values,Check for NULL values, Duplicate values. &emsp;&emsp;&emsp; **Developed By: ROHIT JAIN D**
3. Classify the training data and the test data. &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; **Register No: 21222230120**
4. Calculate the accuracy score, confusion matrix and classification report.
### Program:
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

**Head of the data** &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Null Values:** <br><img width=78%  src="![1](https://github.com/IamShakthi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117913445/a087b6fd-e6f8-4157-8b9b-b860aee8c9ec)">&emsp;<img width=18%  src="![2](https://github.com/IamShakthi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117913445/af9ab0c6-709c-477c-98aa-22a238997ca2)
   "><br><br><br>
**Transformed Data:**&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**X Values:**
<br><img height=10% width=48% src="    ![3](https://github.com/IamShakthi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117913445/e91f658d-3488-4a25-b67c-ca4d3ca3b50c)
       ">&emsp;<img height=10% width=48% src="  ![4](https://github.com/IamShakthi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117913445/7baf445c-f01e-46c6-9563-180e141973d0)
      "><br><br><br>

**Y Values:** &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Y Predicted Values:** <br>
<img src="            ![5](https://github.com/IamShakthi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117913445/a57a2f2a-f08a-41ee-8ead-5638266dd244)
           ">&emsp;<img valign=top src="       ![6](https://github.com/IamShakthi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117913445/2f71d724-7f05-4b84-82e3-f00942b052cc)
        "><br><br><br>
**Accuracy:**&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Confusion Matrix:**&emsp;&emsp;&emsp;&emsp;**Classification Report:**
<br>
<img valign=top src="           ![7](https://github.com/IamShakthi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117913445/281a1dd9-747b-4169-89ef-c56fa719eb86)
         ">&emsp;&emsp;<img valign=top src="           ![8](https://github.com/IamShakthi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117913445/324b69b5-f274-4998-af9c-5aa15c69711b)
       ">&emsp;&emsp;&emsp;&emsp;<img valign=top src="       ![9](https://github.com/IamShakthi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/117913445/14de9f77-5971-4bbc-b998-78df374a106d)
            ">

### Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
