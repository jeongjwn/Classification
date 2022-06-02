#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[11]:


#Read data as dataframe
df = pd.read_csv('../data/vertebral_column_data/column_2C.dat', header=None)
df.columns = ['style']
df1 = df['style'].str.split(' ', expand=True)
df1.columns = ['pelvic incidence','pelvic tilt', 'lumbar lordosis angle', 'sacral slope', 'pelvic radius', 'grade of spondylolisthesis', 'classes']

#Classes as binary
df1 = df1.replace({'NO':'0', 'AB' : '1'})

#convert df to numeric
df1 = df1.apply(pd.to_numeric, errors='ignore')
df1


# Plotting scatterplots of independent variables

# In[13]:


pairplot_fig = sns.pairplot(df1, vars=df1.columns[0:-1], hue = 'classes')
plt.show()


# Plotting boxplots for each independent variables

# In[15]:


fig = plt.figure()
fig.suptitle('Pelvic Incidence', fontweight = 'bold')
sns.boxplot(y = 'pelvic incidence', x = 'classes', data = df1, hue = 'classes')
plt.show()


# In[21]:


fig = plt.figure()
fig.suptitle('Pelvic Tilt', fontweight = 'bold')
sns.boxplot(y = 'pelvic tilt', x = 'classes', data = df1, hue = 'classes')
plt.show()


# In[22]:


fig = plt.figure()
fig.suptitle('Lumbar Lordosis Angle', fontweight = 'bold')
sns.boxplot(y = 'lumbar lordosis angle', x = 'classes', data = df1, hue = 'classes')
plt.show()


# In[23]:


fig = plt.figure()
fig.suptitle('Sacral Slope', fontweight = 'bold')
sns.boxplot(y = 'sacral slope', x = 'classes', data = df1, hue = 'classes')
plt.show()


# In[24]:


fig = plt.figure()
fig.suptitle('Pelvic Radius', fontweight = 'bold')
sns.boxplot(y = 'pelvic radius', x = 'classes', data = df1, hue = 'classes')
plt.show()


# In[25]:


fig = plt.figure()
fig.suptitle('Grade of Spondylolisthesis', fontweight = 'bold')
sns.boxplot(y = 'grade of spondylolisthesis', x = 'classes', data = df1, hue = 'classes')
plt.show()


# Splitting training set: first 80 rows of Class 0 and first 150 rows of class 1

# In[44]:


#Training set
df_cls0 = df1.loc[df1['classes']==0]
df_cls1 = df1.loc[df1['classes']==1]

train_cls0 = df_cls0.iloc[:80]
train_cls1 = df_cls1.iloc[:150]
traindf = pd.concat([train_cls0, train_cls1])

#Testing set
test_cls0 = df_cls0.iloc[80:]
test_cls1 = df_cls1.iloc[150:]
testdf = pd.concat([test_cls0, test_cls1])

traindata = traindf.values
testdata = testdf.values

X_train, y_train = traindata[:, :-1], traindata[:, -1]
X_test, y_test = testdata[:, :-1], testdata[:, -1]


# Practicing Classification using KNN with Euclidean Distance

# In[43]:


clf = KNeighborsClassifier(n_neighbors=4)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


# In[52]:


test_k = []
for i in range(208, 0, -3): 
    test_k.append(i) 
    
error = []
error_train = []
for i in test_k:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    pred_tr_i = knn.predict(X_train)
    error_train.append(np.mean(pred_tr_i != y_train))


error_result= {'k':test_k, 'error': error, 'error_train': error_train}
firstdf = pd.DataFrame(data = error_result)

#Plotting train and test errors in terms of k
plt.plot(firstdf['k'],firstdf['error'],label = "test")
plt.plot(firstdf['k'],firstdf['error_train'],label = "train")
plt.legend()
plt.xlabel("K")
plt.ylabel("error")
plt.title("Euclidean Metric: Train & Test Errors in terms of k")

#Confusion Matrix
y_pred = clf.predict(X_test)
c_matrix = confusion_matrix(y_test, y_pred)
min_error = firstdf['error'].min()
print('Most Suitable k ', firstdf.loc[firstdf['error'] == min_error, 'k'].item())

FP =   c_matrix.item(0,1)
FN = c_matrix.item(1,0)
TP = c_matrix.item(0,0)
TN = c_matrix.item(1,1)
TPR = TP/(TP+FN)
TNR = TN/(TN+FP) 
print('True Positive Rate: ', TPR )
print('True Negative Rate: ', TNR)


# In[49]:


print('Classificationn Report: ', '\n', classification_report(y_test, y_pred))


# In[ ]:




