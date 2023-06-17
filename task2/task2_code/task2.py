#!/usr/bin/env python
# coding: utf-8

# In[23]:


# import necessary libraries
from pycaret.datasets import get_data
from pycaret.classification import *
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import pandas as pd 


# In[24]:


df = pd.read_csv('houseprice11.csv') # read csv dat
df.head()


# In[25]:


## Remove unnecessary columns such as names and dates
df=df.drop(["date","air quality level"],axis=1)
#View column names
# df.columns


# In[26]:


from pycaret.classification import *
# Train the model
clf=setup(data=df,target='class')


# In[27]:
# bonus 优化
# create individual models for stacking
Extra_Trees_Classifier = create_model('et')
# ridge = create_model('ridge')
# lda = create_model('lda')
# gbc = create_model('gbc')
# xgboost = create_model('xgboost')
# stacker = stack_models(estimator_list = [ridge,lda,gbc,xgboost], meta_model = Extra_Trees_Classifier)

# Bagging（Bootstrap聚合）
# dt_boosted = ensemble_model(Extra_Trees_Classifier)
# dt_boosted = ensemble_model(Extra_Trees_Classifier, method='Boosting')

# save_model(Extra_Trees_Classifier, 'task2_model')

# rf_holdout_pred = predict_model(Extra_Trees_Classifier)

# 预测
# new_data = pd.read_csv('Test_Data.csv')
# new_data = new_data.drop(["date","air quality level", "region"],axis=1)

from pycaret.classification import *

# predictions = predict_model(Extra_Trees_Classifier,data=new_data)

# pd.set_option("display.max_columns", None)
# print(predictions)
# print(predictions.iloc[:, -2])
# predictions.to_csv('predictions.csv', index=False, header=True, columns=['Label'])
# predictions.to_csv('predictions.csv')
# predictions.to_csv('predictions.csv', index=False, header=True, columns=['class'])
# y_pred = rf_holdout_pred.predict(new_data)

# apply preprocessing to test data
from pycaret.classification import *


# In[28]:

'''
new_data = pd.read_csv('Test_Data.csv')
## Remove unnecessary columns such as names and dates
new_data = new_data.drop(["date","air quality level",],axis=1)        
#View column names
# new_data.columns


# In[29]:


# Make predictions on data using the loaded model
y_pred = Extra_Trees_Classifier.predict(new_data)


# In[ ]:
'''



