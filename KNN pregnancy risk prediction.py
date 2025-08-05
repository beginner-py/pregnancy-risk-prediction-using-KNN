#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 


# In[66]:


df =  pd.read_csv(r"C:\Users\SASUB\Downloads\archive (1)\Dataset - Updated.csv")


# In[67]:


df.head()


# In[68]:


df.info()


# In[69]:


df.describe()


# In[70]:


df.shape


# In[71]:


df.isnull().sum()


# In[72]:


df = df.dropna(subset=["Risk Level"])


# In[73]:


df.isnull().sum()


# In[74]:


cols = ['Previous Complications', 'Preexisting Diabetes']

for col in cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(0)


# In[75]:


cols_missing = ['Systolic BP', 'Diastolic', 'BS', 'Heart Rate', 'BMI']
for col in cols_missing:
    df[col] = df[col].fillna(df.groupby('Risk Level')[col].transform('mean'))


# In[76]:


df.isnull().sum()


# In[77]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[78]:


X = df.drop('Risk Level', axis=1)
le = LabelEncoder()
y = le.fit_transform(df['Risk Level'])


# In[79]:


print(y)


# In[80]:


pd.Series(y).value_counts()


# In[81]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[82]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[87]:


knn = KNeighborsClassifier(n_neighbors = 3)  
knn.fit(X_train, y_train)


# In[88]:


y_train_pred = knn.predict(X_train)

# Evaluation Metrics
print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))
print("\nClassification Report:\n", classification_report(y_train, y_train_pred))
print("\nAccuracy Score:", accuracy_score(y_train, y_train_pred))


# In[89]:


y_pred = knn.predict(X_test)

# Evaluation Metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))


# In[90]:


import pickle

# Save Model
with open('prediction.pkl', 'wb') as model_file:
    pickle.dump(knn, model_file)

# Save Scaler
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("prediction.pkl and scaler.pkl saved successfully!")


# In[ ]:





# In[ ]:




