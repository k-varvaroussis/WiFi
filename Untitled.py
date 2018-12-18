
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from numpy import random
from matplotlib import pyplot as plt
from pylab import *  # for figsize
import matplotlib.dates as mdates
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
import seaborn as sns
sns.set()
sns.set(style="ticks")

import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in(username = 'k_varvaroussis', api_key = 'oXOw5WhzVgkZYfA4H8VK')

import sklearn as sl
import sklearn.linear_model
import sklearn.datasets
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import io
import requests
import os
import tarfile
from six.moves import urllib
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#from future_encoders import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error

from IPython.core.display import display,HTML

from IPython.display import HTML
import warnings

warnings.filterwarnings("ignore")

import io
import requests


# In[ ]:


#print(sklearn.__version__)


# ## Load training and validation set

# In[ ]:


url="https://raw.githubusercontent.com/k-varvaroussis/WiFi/master/trainingData.csv"
s=requests.get(url).content
train = pd.read_csv(io.StringIO(s.decode('utf-8')))


# In[ ]:


url2="https://raw.githubusercontent.com/k-varvaroussis/WiFi/master/validationData.csv"
s2=requests.get(url2).content
valid = pd.read_csv(io.StringIO(s2.decode('utf-8')))


# In[ ]:


pd.set_option('display.max_columns', 90)


# In[ ]:


#train.head()


# In[ ]:


#train.describe()


# # Preprocessing

# ## Drop duplicates (unused)

# In[ ]:


train = train.drop_duplicates(subset = None, keep = "first", inplace = False)


# In[ ]:


#train.info()


# ## Change unix-time into datetime-format

# In[ ]:


train.TIMESTAMP = pd.to_datetime(train.TIMESTAMP, unit = 's')


# In[ ]:


valid.TIMESTAMP = pd.to_datetime(valid.TIMESTAMP, unit = 's')


# In[ ]:


#valid.head(20)


# ## Transform 100s into -105s

# In[ ]:


waps2 = waps.replace(100,-105)


# In[ ]:


#waps3.info()


# In[ ]:


#train2.info()


# In[ ]:


keeps = ["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID", "SPACEID","RELATIVEPOSITION","USERID","PHONEID"]


# In[ ]:


train3 = train2[keeps]


# In[ ]:


train4 = pd.concat([train3, waps2], axis=1)


# In[ ]:


#train4.head()


# ## NANs, 0-variation, Outliers

# In[ ]:


tr_descr = train.describe()


# In[ ]:


#tr_descr


# In[ ]:


#train.info()


# In[ ]:


#train.dtypes


# In[ ]:


#valid.describe()


# ### Check for NANs

# In[ ]:


train.isnull().values.any()


# In[ ]:


train.isna().values.any()


# ### remove WAPs with only 100s

# In[ ]:


#type(train.std())


# In[ ]:


train2 = train[train.std()[train.std()!=0].index]


# In[ ]:


#train2.head()


# ### Look for outliers

# In[ ]:


#train2.describe()


# ### Check if there are zeros in the WAPs

# In[ ]:


waps = train2.drop(columns = ["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID", "SPACEID","RELATIVEPOSITION","USERID","PHONEID"])


# In[ ]:


#waps.head()


# In[ ]:


0 in waps.values


# ### Test space ID has only integers

# In[ ]:


#train4.SPACEID.describe()


# In[ ]:


np.array_equal(train4.SPACEID, train4.SPACEID.astype(int))


# ### Apply new range (-30 : -85)

# In[ ]:


#X_range = X_train2 #This needs the X_train2 formed below at "X,y Split"
#
#count_cols = {}
#for i in X_train2.columns:
#    count_cols[i] = 0
#
#count_rows = {}
#for i in X_train2.index:
#    count_rows[i] = 0
#
#for i in X_train2.columns:
#    for j in X_train2.index:
#        if ((X_range.loc[[j],[str(i)]] < -40).bool() & (X_range.loc[[j],[str(i)]] > -85).bool()):
#            count_cols[i] += 1
#            count_rows[j] += 1
#
#new_cols = [col for col in count_cols if count_cols[col]!=0]
#new_rows = [row for row in count_rows if count_rows[row]!=0]#


# In[ ]:


#df_ranged = X_train2[new_cols]
#df_ranged.index = new_rows


# ### create new_range function

# In[ ]:


def new_range(lower_limit, upper_limit, old_df): # input (sliced) df which consists only of columns to be ranged

    df_ranged = old_df                           # returns df where all columns or rows are deleted that contain 
                                                 # not one single value within the limits
    count_cols = {}
    for i in old_df.columns:
        count_cols[i] = 0
    
    count_rows = {}
    for i in old_df.index:
        count_rows[i] = 0
    
    for i in old_df.columns:
        for j in old_df.index:
            if ((df_ranged.loc[[j],[str(i)]] < -40).bool() & (df_ranged.loc[[j],[str(i)]] > -85).bool()):
                count_cols[i] += 1
                count_rows[j] += 1
    
    new_cols = [col for col in count_cols if count_cols[col]!=0]
    new_rows = [row for row in count_rows if count_rows[row]!=0]
    
    df_ranged = old_df[new_cols]
    df_ranged.index = new_rows
    
    return df_ranged


# In[ ]:


new_range(-80,-40, X_train2)


# ## Visualizations

# ### Histograms of L & L

# In[ ]:


#train4.LATITUDE.describe()


# In[ ]:


bins = arange(4.864746e+06,4.86520e+06,0.00002e+06)
#plt.hist(train4.LATITUDE,bins = bins)


# In[ ]:


#train4.LONGITUDE.describe()


# In[ ]:


bins = arange(-7691.338400,-7301,20)
#plt.hist(train4.LONGITUDE,bins = bins)


# In[ ]:


#plt.style.use('seaborn-whitegrid')


# ### Form predictions arrays into df's

# In[5]:


x = 1


# In[6]:


get_ipython().run_line_magic('who', '%run #% cell magic')


# In[ ]:


valid_pred1 = pd.DataFrame(data = valid_pred1[0:,0:],          # sections below needed
                           index = range(0,len(valid_pred1)),
                           columns = ['LONGITUDE','LATITUDE'])


# In[ ]:


y_3 = pd.DataFrame(data = y_3[0:,0:],
                           index = range(0,len(y_3)),
                           columns = ['BUILDING','FLOOR'])


# In[ ]:


#y_3


# In[ ]:


len(valid_pred1[1:,0])


# # 1. multi-output classification

# ## Splits

# ### first-order split

# In[ ]:


Ssplit = SSS(n_splits=1, test_size=0.2, random_state=42)
Ssplit
for train_index, test_index in Ssplit.split(train4,train4["LONGITUDE"]):
    train_set = train4.loc[train_index]
    test_set = train4.loc[test_index]


# In[ ]:


train_set.isna().values.any()


# In[ ]:


train_set.info()


# ### X,y Split

# In[ ]:


X_train = train_set.drop(columns =["BUILDINGID","FLOOR", "LONGITUDE", "LATITUDE","SPACEID", "RELATIVEPOSITION","PHONEID","USERID"])


# In[ ]:


#X_train.columns


# In[ ]:


X_train2 = X_train.loc[:,common_cols(waps7,X_train)].dropna(axis = 'columns', thresh = 1)


# In[ ]:


#X_train2.info()


# In[ ]:


X_train2.isnull().values.any()


# In[ ]:


keeps = ["BUILDINGID","FLOOR"]
y_train = train_set[keeps]


# In[ ]:


y_train.isnull().values.any()


# ## Self-made accuracy

# ### Define accuracy measure

# In[ ]:


#train_set.head()


# In[ ]:


def acc_cust(test,pred):  # first paramter has to be a df with columns BUILDINGID and FLOOR, second array
    
    correct = 0
    almost = 0
    
    build_id = list(i for i in test["BUILDINGID"])
    floor = list(i for i in test["FLOOR"])
    
    for i in range(0,len(pred)):
        if (build_id[i] == pred[i][0]) & (floor[i] == pred[i][1]):
            correct += 1
        elif (build_id[i] == pred[i][0]) & ((floor[i] - pred[i][1]) == 1): #one floor difference is 'almost' correct
            almost += 1
    
    result = [correct / len(pred), almost / len(pred)]
    
    return result


# ### test accuracy measure

# In[ ]:


dic1 = {'BUILDINGID':[0,1,2,1], 'FLOOR':[1,2,3,4]}
test = pd.DataFrame(data = dic1)


# In[ ]:


#dic2 = {'BUILDINGID':[0,2,0,1], 'FLOOR':[1,2,3,4]}
#pred = pd.DataFrame(data = dic2)


# In[ ]:


pred = [[0,1],[1,1],[1,3],[1,1]]


# In[ ]:


#test.loc[[1],["BUILDINGID"]] == pred.loc[[1],["BUILDINGID"]])#.loc[0].bool() & (test.loc[[0],["FLOOR"]] == pred.loc[[0],["FLOOR"]]).loc[0].bool()


# In[ ]:


#test.loc[[0],["BUILDINGID"]] #test.loc[[0],["BUILDINGID"]] #== pred.loc[[0],["BUILDINGID"]] & False


# In[ ]:


print(acc_cust(test, pred))#works


# ## Decision Tree

# In[ ]:


regr_1 = DecisionTreeClassifier(max_depth=2, random_state = 27)
regr_2 = DecisionTreeClassifier(max_depth=5, random_state = 27)
regr_3 = DecisionTreeClassifier(max_depth=8, random_state = 27)


# ### Cross-validation

# In[ ]:


#skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=26) DOES NOT WORK ON MULTI_LABEL


# In[ ]:


kf = KFold(n_splits = 3, random_state = 27)


# In[ ]:


#train_set.head()


# ### Train

# In[ ]:


indices = {}
i = 0
for train_indices, test_indices in kf.split(X):
    i = i + 1
    
    #indices[i]=train_indices 
    X_tr = X.loc[train_indices].dropna()
    y_tr = y.loc[train_indices].dropna()
    
    regr_1.fit(X_tr,
                y_tr)
    regr_2.fit(X_tr,
                y_tr)
    regr_3.fit(X_tr,
                y_tr)
    
    y_1 = regr_1.predict(X.loc[test_indices].dropna())
    y_2 = regr_2.predict(X.loc[test_indices].dropna())
    y_3 = regr_3.predict(X.loc[test_indices].dropna())
    
    #print(np.nonzero(regr_1.feature_importances_))
    #print(np.nonzero(regr_2.feature_importances_))
    #print(np.nonzero(regr_3.feature_importances_))
    #print(str(i) +". fold")
    #print("\n")
#
    #print("accuracy measure on cv-test-set (max. depth = 2)")
    #print(acc_cust(train_set.dropna(),y_1))
    #print("\n")
    #
    #print("accuracy measure on cv-test-set (max. depth = 5)")
    #print(acc_cust(train_set.dropna(),y_2))
    #print("\n")
    #
    #print("accuracy measure on cv-test-set (max. depth = 8)")
    #print(acc_cust(train_set.dropna(),y_3))
    #    


# ### Feature selection

# ### Visualize predictions

# In[ ]:


measured = {                              # run sections below
  "x":y_valid2['LONGITUDE'], 
  "y": y_valid2['LATITUDE'], 
  "z": valid4['FLOOR'], 
  "marker": {
    "color": "blue", 
    "colorscale": "Viridis", 
    "opacity": 1, 
    "size": 4
  }, 
  "mode": "markers", 
  "name": "measured", 
  "type": "scatter3d"
}


# In[ ]:


predictions = {
  "x":valid_pred1['LONGITUDE'], 
  "y": valid_pred1['LATITUDE'], 
  "z": y_3['FLOOR'], 
  "marker": {
    "color": "yellow", 
    "colorscale": "Viridis", 
    "opacity": 1, 
    "size": 4
  }, 
  "mode": "markers", 
  "name": "predictions", 
  "type": "scatter3d"
}


# In[ ]:


data = Data([measured,predictions])
layout = {
  "scene": {
    "xaxis": {"title": "Longitude"}, 
    "yaxis": {"title": "Atitude"}, 
    "zaxis": {"title": "Floor"}
  }, 
  "title": "WiFi-Locationing in 3 Buildings"
}


# In[ ]:


fig = Figure(data=data, layout=layout)
py.iplot(fig)


# function returning ordered list of most important attributes, can be used for classifiers as well as regression problems

# In[ ]:


def impurity_decrease(data, target, maximum_tree_depth=8):
    
    dt = tree.DecisionTreeClassifier(max_depth=maximum_tree_depth) #here we can change from classifier to regression

    dt = dt.fit(data, target)
    
    dt = dt.feature_importances_        
    
    column = data.columns
    
    column = column.to_series()
    
    dt_features = pd.DataFrame()        
    
    dt_features["Attribute"] = column
    
    dt_features["Feature Importance"] = dt
    
    dt_features = dt_features.sort_values(by=["Feature Importance"], ascending=False)        
    
    return dt_features


# In[ ]:


#impurity_decrease(X,y).head(10)


# ## Random forest 

# only keep important attributes: [ not used]

# In[ ]:


#keeps = list(impurity_decrease(X_train2,y_train).head(30).Attribute)


# ### Form test_set inputs

# In[ ]:


X_test = test_set.drop(columns =["BUILDINGID","FLOOR", "LONGITUDE", "LATITUDE","SPACEID", "RELATIVEPOSITION","PHONEID","USERID"])


# In[ ]:


X_test2 = X_test.loc[:,common_cols(waps7,X_test)].dropna(axis = 'columns', thresh = 1)


# In[ ]:


keeps = ["BUILDINGID","FLOOR"]
y_test = test_set[keeps]


# ### Non-CV fit & prediction

# In[ ]:


clf1 = RandomForestClassifier(n_estimators=1500, max_depth=80,
                             random_state=1991)
#clf2 = RandomForestClassifier(n_estimators=100, max_depth=2,
#                             random_state=0)
#clf3 = RandomForestClassifier(n_estimators=100, max_depth=2,
#                             random_state=0)


# In[ ]:


model = clf1.fit(X_train2,y_train)

y_1 = model.predict(X_test2)
print(acc_cust(test_set,y_1))


# In[ ]:


i = 0
#for train_indices, test_indices in kf.split(X):
#    i = i + 1
#    
#    #indices[i]=train_indices 
#    X_tr = X.loc[train_indices].dropna()
#    y_tr = y.loc[train_indices].dropna()
#    
#    clf1.fit(X_tr,
#                y_tr)
#    #regr_2.fit(X_tr,
#    #            y_tr)
#    #regr_3.fit(X_tr,
#    #            y_tr)
#    
#    y_1 = clf1.predict(X.loc[test_indices].dropna())
#    #y_2 = regr_2.predict(X.loc[test_indices].dropna())
#    #y_3 = regr_3.predict(X.loc[test_indices].dropna())
#    
#    #print(X_tr)
#    #print("\n") #for checking on dataframes
#    #print(y_1)
#    
#    print(str(i) +". fold")
#    print("\n")
#    
#    print("accuracy measure on cv-test-set (max. depth = 3)")
#    print(acc_cust(train_set.dropna(),y_1))
#    print("\n")
#    
#    print("accuracy measure on cv-test-set (max. depth = 5)")
#    print(acc_cust(train_set.dropna(),y_2))
#    print("\n")
#    
#    print("accuracy measure on cv-test-set (max. depth = 8)")
#    print(acc_cust(train_set.dropna(),y_3))
    
    


# ## Ada Boost

# In[ ]:


clf2 = AdaBoostClassifier()
multi_target_classifier = MultiOutputClassifier(clf2, n_jobs=-1)
multi_target_classifier.fit(X_train2,y_train)
preds = multi_target_classifier.predict(X_test2)
print(acc_cust(test_set,preds))


# ## GBT

# In[ ]:


clf3 = GradientBoostingClassifier()
multi_target_classifier2 = MultiOutputClassifier(clf3, n_jobs=-1)
clf_model3 = multi_target_classifier2.fit(X_train2,y_train)
preds2 = clf_model3.predict(X_test2)
print(acc_cust(test_set,preds2))


# #  1. multi-output regression 

# This is the continuation of the baseline multi-output classification model, from which we take the output as the input of the following regression model to predict longitude & latitude.

# ## Append cls predictions to previous test set (unused)

# In[ ]:


test_set["Floor"] = list(y_1[i][1] for i in range(0,len(y_1)))


# In[ ]:


test_set["Building_id"] = list(y_1[i][0] for i in range(0,len(y_1)))


# ## Form regression label

# In[ ]:


keeps = ["LONGITUDE","LATITUDE"]
reg_y_train = train_set[keeps]
reg_y_test = test_set[keeps]


# ## Random Forest

# In[ ]:


reg_4 = RandomForestRegressor(n_estimators = 500, max_depth = 20, random_state = 1991)


# In[ ]:


reg_model1 = reg_4.fit(X_train2,reg_y_train)


# In[ ]:


reg_model1.score(X_test2,reg_y_test)


# In[ ]:


#reg_model1


# ## GBT

# In[ ]:


reg_model2 = MultiOutputRegressor(GradientBoostingRegressor(), n_jobs=-1).fit(X_train2,reg_y_train)


# In[ ]:


reg_model2.score(X_test2, reg_y_test)


# # Solely floor classification

# Only use floor as independent variable to increase accuracy

# ## Form variables

# In[ ]:


keeps = ["FLOOR"]
cls_y_train = train_set[keeps]
cls_y_test = test_set[keeps]


# In[ ]:


X_test2 = test_set.drop(columns =["FLOOR", "LONGITUDE", "LATITUDE","SPACEID", "RELATIVEPOSITION","PHONEID","USERID"])


# In[ ]:


X_train2 = train_set.drop(columns =["FLOOR", "LONGITUDE", "LATITUDE","SPACEID", "RELATIVEPOSITION","PHONEID","USERID"])


# In[ ]:


train_set.info()


# ## Train & Model

# In[ ]:


clf2 = RandomForestClassifier(n_estimators=600, max_depth=20,
                             random_state=1991)


# In[ ]:


cls_model2 = clf2.fit(X_train2,cls_y_train)

#y_1 = model2.predict(X_test)
cls_model2.score(X_test2,cls_y_test)


# # Test models on validation set

# ## Preprocess validation set

# In[ ]:


#valid.info()


# In[ ]:


valid.isnull().values.any()


# In[ ]:


train.isna().values.any()


# In[ ]:


valid2 = valid[valid.std()[valid.std()!=0].index]


# In[ ]:


valid2.info()


# In[ ]:


waps6 = valid2.drop(columns = ["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID","PHONEID"])


# In[ ]:


0 in waps6.values


# In[ ]:


waps7 = waps6.replace(100,-105)


# In[ ]:


keeps = ["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID","PHONEID"]


# In[ ]:


valid3 = valid2[keeps]


# In[ ]:


valid4 = pd.concat([valid3, waps7], axis=1)


# In[ ]:


valid5 = valid4.loc[:,common_cols(X_train2,valid4)].dropna(axis = 'columns', thresh = 1)


# In[ ]:


#len(waps8.columns)


# ### Find common varying waps in train & valid

# In[ ]:


def common_cols(less_colsDF, df):
    com_cols = []
    for i in range(0,len(less_colsDF.columns)):
        if any(elem in list(less_colsDF.columns[i]) for elem in list(df.columns)):
            com_cols.append(less_colsDF.columns[i])
    return com_cols


# In[ ]:


#len(valid4.columns)


# ## Classification

# In[ ]:


keeps = ["BUILDINGID", "FLOOR"]
y_valid = valid4[keeps]


# In[ ]:


waps8 = waps7[common_cols(waps7,X_train2)]# validation-waps with the same columns as training set


# In[ ]:


waps9 = waps8.loc[:,common_cols(X_train2,valid4)].dropna(axis = 'columns', thresh = 1)


# In[ ]:


#len(valid5.columns)


# ### Random Forest

# In[ ]:


y_3 = model.predict(waps9)
print(acc_cust(valid4,y_3))


# ### GBT

# In[ ]:


y_4 = clf_model3.predict(waps9)
print(acc_cust(valid4,y_4))


# ## Regression

# ### Define regression_y

# In[ ]:


keeps = ["LATITUDE", "LONGITUDE"]
y_valid = valid4[keeps]


# ### Use same column order

# In[ ]:


cols = y_valid.columns.tolist()


# In[ ]:


cols = cols[-1:] + cols[:-1]


# In[ ]:


cols


# In[ ]:


y_valid2 = y_valid[cols]


# ### Calculate R-squared

# In[ ]:


valid_pred1 = reg_model1.predict(valid5) # Random Forest


# In[ ]:


print(r2_score(y_valid2, valid_pred1)) # Random Forest 


# In[ ]:


#reg_model2.score(valid5, y_valid2)    # GBT


# In[ ]:


valid_pred2 = reg_model2.predict(valid5)  # GBT


# In[ ]:


print(r2_score(y_valid2, valid_pred2))     # GBT


# ### Calculate RMSE

# In[ ]:


print(mean_squared_error(y_valid2, valid_pred1)**0.5) # Random Forest


# # The product & the process

# ## Goal of the client

# - System (App) to help people navigate in complex, unfamiliar interior space 

# ## Data-scientific requierements for development

# ### Basic version

# - Collect training data for every new building
# - Embedd trained model into app
# - Use incoming streams of user data for further refinement

# ### Future version

# - Collect Altitude, combine with Latitude & Longitude
# - Combine 3D-locationing with 3D-world-map

# ## Our Goal 

# ### Data Science processes for different algorithms consisting of different pipelines:

# - Preprocess (Change data-types, remove NANs, outliers, alter ranges, normalize...)
# 
# - Feature Selection (WAPs with variance, Random-Forest-based, stationarity)
# 
# - Stratified Splitting (training-, test- & validation set)
# 
# - Cross-validation
# 
# - Testing on test set
# 
# - paramter-tuning based on test set
# 
# - test on validation set

# ### Choose model (algorithm) based on validation set metrics

# ### Automate process for incoming streams of data

# ## Progress

# ### Done:

# - The idea of predicting the location with the given data, works
# - The next improvement will be in applying new range
# - -> Preprocessing, esp. Router selection important
# - Multi-output regression & classification works for many models
# - -> reduces processing time when choosing algorithms automatically and when automating for streams
# 
# 

# ### The next steps 

# - Applying new range 
# - Test more algorithms
# - Look at user behaviour (external validity)
# - Different phones, different models?
# - build pipelines, generalize functions & build packages
# - Are labels of Classification predictors of regression in other models?

# ## Results

# In[ ]:


### Visualize predictions

measured = {                              # run sections below
  "x":y_valid2['LONGITUDE'], 
  "y": y_valid2['LATITUDE'], 
  "z": valid4['FLOOR'], 
  "marker": {
    "color": "blue", 
    "colorscale": "Viridis", 
    "opacity": 1, 
    "size": 4
  }, 
  "mode": "markers", 
  "name": "measured", 
  "type": "scatter3d"
}

predictions = {
  "x":valid_pred1['LONGITUDE'], 
  "y": valid_pred1['LATITUDE'], 
  "z": y_3['FLOOR'], 
  "marker": {
    "color": "yellow", 
    "colorscale": "Viridis", 
    "opacity": 1, 
    "size": 4
  }, 
  "mode": "markers", 
  "name": "predictions", 
  "type": "scatter3d"
}

data = Data([measured,predictions])
layout = {
  "scene": {
    "xaxis": {"title": "Longitude"}, 
    "yaxis": {"title": "Atitude"}, 
    "zaxis": {"title": "Floor"}
  }, 
  "title": "WiFi-Locationing in 3 Buildings"
}

fig = Figure(data=data, layout=layout)
py.iplot(fig)


# # Reflection on done work

# ## Used Multi-output methods
