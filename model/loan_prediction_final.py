#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('loan_tree.csv')
df


# In[3]:




a = {'active': 1, 'nonactive': 0}
df['ACTIVE/NON'] = df['ACTIVE/NON'].map(a)
df


# In[17]:


X = df[['AGE','annual_income','ACTIVE/NON','CREDIT SCORE','Credit_history','loan amount']]
Y = df['status']
print(X)


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25)


# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(x_train, y_train)

LogisticRegression()



import pickle
pickle.dump(model, open('loan_prediction_final.pkl','wb') )
model=pickle.load(open('loan_prediction_final.pkl','rb'))


# In[22]:


pred = model.predict(x_test)
accuracy_score(y_test,pred)




from joblib import dump, load
dump(model,'loan_prediction_final.joblib')


# In[28]:


"""import pickle
pickle.dump(model, open('loan_prediction_final','wb') )
model=pickle.load(open('loan_prediction_final','rb'))
print(model)


# In[25]:


print(pred)
print(x_test)
print(y_test)


# In[29]:


output = model.predict([[45,200000,1,400,1,500000]])
print(output)"""


# In[ ]:




