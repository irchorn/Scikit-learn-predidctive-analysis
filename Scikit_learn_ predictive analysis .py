#!/usr/bin/env python
# coding: utf-8

# # IS 362 PROJECT 4

# In[1]:


import pandas as pd



from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier


from sklearn.neighbors import KNeighborsClassifier


from sklearn.naive_bayes import GaussianNB


# In[ ]:





# <b>Create a pandas DataFrame with a subset of the columns in the dataset.</b>

# In[2]:


filename_mushrooms = 'agaricus-lepiota.data'
df_mushrooms = pd.read_csv(filename_mushrooms)
display(df_mushrooms.head())


# In[3]:


df_mushrooms.columns


# In[4]:


df_mushrooms.columns =['class', 'cap-shape', 'cap-syrface', 'cap-color', 'bruises?', 'odor', 'gill-attachment', 'gill-spacing',
             'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
             'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
            'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']


# In[5]:


df_mushrooms.columns


# In[6]:


df_mushrooms


# In[7]:


df_subset_mushrooms = df_mushrooms[['class', 'odor', 'habitat']]


# In[8]:


df_subset_mushrooms


#  <b>Replace the codes used in the data with numeric values</b>

# In[9]:


labelEncoder = LabelEncoder()
df_subset_mushrooms_n = df_subset_mushrooms.apply(labelEncoder.fit_transform)


# In[13]:


df_subset_mushrooms_n


# <b>Exploratory data analysis.</b>
# Logistic Regression Classification: Odor has the highest test accuracy

# In[14]:


from sklearn.model_selection import train_test_split
y = df_subset_mushrooms_n["class"].values    # "class" column as numpy array.
x = df_subset_mushrooms_n.drop(["class"], axis=1).values    # All data except "class" column. 
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.2)    # Split data for train and test.


# In[15]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="lbfgs")
lr.fit(x_train,y_train)
print("Test Accuracy: {}%".format(round(lr.score(x_test,y_test)*100,2)))


# In[16]:


from sklearn.model_selection import train_test_split
y = df_subset_mushrooms_n["odor"].values    
x = df_subset_mushrooms_n.drop(["odor"], axis=1).values    
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.2)    


# In[17]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="lbfgs")
lr.fit(x_train,y_train)
print("Test Accuracy: {}%".format(round(lr.score(x_test,y_test)*100,2)))


# In[18]:


from sklearn.model_selection import train_test_split
y = df_subset_mushrooms_n["habitat"].values   
x = df_subset_mushrooms_n.drop(["habitat"], axis=1).values    
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.2)   


# In[75]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="lbfgs")
lr.fit(x_train,y_train)
print("Test Accuracy: {}%".format(round(lr.score(x_test,y_test)*100,2)))


# <b>KNN Classification method. Class has the highest test accuracy</b>

# <b>Class</b>

# In[19]:


from sklearn.model_selection import train_test_split
y = df_subset_mushrooms_n["class"].values    # "class" column as numpy array.
x = df_subset_mushrooms_n.drop(["class"], axis=1).values    # All data except "class" column. 
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.2)    # Split data for train and test.


# In[20]:


from sklearn.neighbors import KNeighborsClassifier
best_Kvalue = 0
best_score = 0
for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    if knn.score(x_test,y_test) > best_score:
        best_score = knn.score(x_train,y_train)
        best_Kvalue = i
print("""Best KNN Value: {}
Test Accuracy: {}%""".format(best_Kvalue, round(best_score*100,2)))


# <b>Odor</b>

# In[21]:


from sklearn.model_selection import train_test_split
y = df_subset_mushrooms_n["odor"].values    
x = df_subset_mushrooms_n.drop(["odor"], axis=1).values    
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.2)  


# In[22]:


from sklearn.neighbors import KNeighborsClassifier
best_Kvalue = 0
best_score = 0
for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    if knn.score(x_test,y_test) > best_score:
        best_score = knn.score(x_train,y_train)
        best_Kvalue = i
print("""Best KNN Value: {}
Test Accuracy: {}%""".format(best_Kvalue, round(best_score*100,2)))


# <b>Habitat</b>

# In[23]:


from sklearn.model_selection import train_test_split
y = df_subset_mushrooms_n["habitat"].values   
x = df_subset_mushrooms_n.drop(["habitat"], axis=1).values    
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.2)   


# In[82]:


from sklearn.neighbors import KNeighborsClassifier
best_Kvalue = 0
best_score = 0
for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    if knn.score(x_test,y_test) > best_score:
        best_score = knn.score(x_train,y_train)
        best_Kvalue = i
print("""Best KNN Value: {}
Test Accuracy: {}%""".format(best_Kvalue, round(best_score*100,2)))


# <b>Naive Bayes Classification. Class provides the highest test accuracy</b>

# <b>Class</b>

# In[24]:


from sklearn.model_selection import train_test_split
y = df_subset_mushrooms_n["class"].values    # "class" column as numpy array.
x = df_subset_mushrooms_n.drop(["class"], axis=1).values    # All data except "class" column. 
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.2)    # Split data for train and test.


# In[25]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
print("Test Accuracy: {}%".format(round(nb.score(x_test,y_test)*100,2)))


# <b>Odor</b>

# In[26]:


from sklearn.model_selection import train_test_split
y = df_subset_mushrooms_n["odor"].values    
x = df_subset_mushrooms_n.drop(["odor"], axis=1).values    
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.2)  


# In[27]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
print("Test Accuracy: {}%".format(round(nb.score(x_test,y_test)*100,2)))


# <b>Habitat</b>

# In[28]:


from sklearn.model_selection import train_test_split
y = df_subset_mushrooms_n["habitat"].values   
x = df_subset_mushrooms_n.drop(["habitat"], axis=1).values    
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.2)  


# In[29]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
print("Test Accuracy: {}%".format(round(nb.score(x_test,y_test)*100,2)))


# In[ ]:




