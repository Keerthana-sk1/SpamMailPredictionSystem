import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Loading dataset
raw_mail_data=pd.read_csv('/content/mail_data.csv')
print(raw_mail_data)
#Replacing null values with null string
mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),'')
mail_data.head()
mail_data.shape
#Label Encoding - spam 0, ham 1
mail_data.loc[mail_data['Category']  == 'spam','Category',]=0
mail_data.loc[mail_data['Category']  == 'ham','Category',]=1
#seperate data as text and labels
x=mail_data['Message']
y=mail_data['Category']
print(x)
print(y)
#Splitting data to training data and test data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=3)
print(x.shape)
print(x_train.shape)
print(x_test.shape)
#Feature Extraction - Convert test data to numerical values
feature_extraction=TfidfVectorizer(min_df=1,stop_words='english', lowercase=True)
x_train_features=feature_extraction.fit_transform(x_train)
x_test_features=feature_extraction.transform(x_test)
#converting y values to integers
y_train=y_train.astype('int')
y_test=y_test.astype('int')
print(x_train_features)
#Training the model
model = LogisticRegression()
#training this model with training data
model.fit(x_train_features,y_train)
#Evaluating trained model 
#prediction on training data
prediction_on_training_data=model.predict(x_train_features)
accuracy_on_training_data=accuracy_score(y_train,prediction_on_training_data)
print('Accuracy_on_training_data : ',accuracy_on_training_data)
prediction_on_test_data=model.predict(x_test_features)
accuracy_on_test_data=accuracy_score(y_test,prediction_on_test_data)
print('Accuracy_on_test_data : ',accuracy_on_test_data)
#Building a predictive system
input_mail=["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]
#convert text to feature vectors
input_data_features=feature_extraction.transform(input_mail)
#making prediction
prediction=model.predict(input_data_features)
print(prediction)
if(prediction[0]==1):
  print("Ham mail")
else:
  print("Spam mail")
