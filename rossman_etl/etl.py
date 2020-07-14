import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split

def str_to_date(date):
    return datetime.strptime(date, '%Y-%m-%d').date()
df_train = pd.read_csv("train.csv",sep=',', parse_dates=['Date'],
                    date_parser=str_to_date, low_memory = False)
df_train=df_train.drop(df_train[(df_train.Open == 0) & (df_train.Sales == 0)].index)

df_store = pd.read_csv("store.csv")
df_store.CompetitionOpenSinceMonth.fillna(0, inplace = True)
df_store.CompetitionOpenSinceYear.fillna(0,inplace=True)
df_store.Promo2SinceWeek.fillna(0,inplace=True)
df_store.Promo2SinceYear.fillna(0,inplace=True)
df_store.PromoInterval.fillna(0,inplace=True)
df_store['CompetitionDistance'].fillna(df_store['CompetitionDistance'].median(), inplace = True)



df_train_store = pd.merge(df_train, df_store, how = 'left', on = 'Store')
df_train_store['CompetitionOpenSince'] = np.where((df_train_store['CompetitionOpenSinceMonth']==0) & (df_train_store['CompetitionOpenSinceYear']==0) , 0,(df_train_store.Date.dt.month - df_train_store.CompetitionOpenSinceMonth) + 
                                       (12 * (df_train_store.Date.dt.year - df_train_store.CompetitionOpenSinceYear)) )
df_train_store['SalesperCustomer']=df_train_store['Sales']/df_train_store['Customers']
df_train_store["is_holiday_state"] = df_train_store['StateHoliday'].map({"0": 0, "a": 1, "b": 1, "c": 1})
df_train_store['dayofyear_cos'] = np.cos(2 * np.pi * df_train_store.Date.dt.dayofyear/max(df_train_store.Date.dt.dayofyear))
df_train_store['dayofyear_sin'] = np.sin(2 * np.pi * df_train_store.Date.dt.dayofyear/max(df_train_store.Date.dt.dayofyear))
df_train_store['weekofyear_cos'] = np.cos(2 * np.pi *df_train_store.Date.dt.weekofyear/max(df_train_store.Date.dt.weekofyear))
df_train_store['weekofyear_sin'] = np.sin(2 * np.pi *df_train_store.Date.dt.weekofyear/max(df_train_store.Date.dt.weekofyear))

df_train_store['Year']=df_train_store.Date.dt.year
df_train_store['Year_cos'] = np.cos(2 * np.pi *df_train_store.Date.dt.year/max(df_train_store.Date.dt.year))
df_train_store['Year_sin'] = np.sin(2 * np.pi *df_train_store.Date.dt.year/max(df_train_store.Date.dt.year))

df_train_store['Day']=df_train_store.Date.dt.day
df_train_store['Month']=df_train_store.Date.dt.month
df_train_store['Day_cos'] = np.cos(2 * np.pi *df_train_store.Date.dt.day/max(df_train_store.Date.dt.day))
df_train_store['Day_sin'] = np.sin(2 * np.pi *df_train_store.Date.dt.day/max(df_train_store.Date.dt.day))
df_train_store['month_cos'] = np.cos(2 * np.pi *df_train_store.Date.dt.month/max(df_train_store.Date.dt.month))
df_train_store['month_sin'] = np.sin(2 * np.pi *df_train_store.Date.dt.month/max(df_train_store.Date.dt.month))

del df_train_store['Promo2SinceWeek']
del df_train_store['Promo2SinceYear']
del df_train_store['StateHoliday']
del df_train_store['CompetitionOpenSinceYear']
del df_train_store['CompetitionOpenSinceMonth']
del df_train_store['Date']



df_train_store=pd.get_dummies(df_train_store, columns=["Assortment", "StoreType","PromoInterval"], prefix=["is_Assortment", "is_StoreType","is_PromoInteval"])











df_test = pd.read_csv("test.csv",sep=',', parse_dates=['Date']
                       , date_parser=str_to_date,
                       low_memory = False)
df_test.fillna(1, inplace = True)
df_test_store = pd.merge(df_test, df_store, how = 'left', on = 'Store')
df_test_store['StateHoliday'] = df_test_store['StateHoliday'].astype('category')
df_test_store['Assortment'] = df_test_store['Assortment'].astype('category')
df_test_store['StoreType'] = df_test_store['StoreType'].astype('category')
df_test_store['PromoInterval']= df_test_store['PromoInterval'].astype('category')
df_test_store['StateHoliday_cat'] = df_test_store['StateHoliday'].cat.codes
df_test_store['Assortment_cat'] = df_test_store['Assortment'].cat.codes
df_test_store['StoreType_cat'] = df_test_store['StoreType'].cat.codes
df_test_store['PromoInterval_cat'] = df_test_store['PromoInterval'].cat.codes
df_test_store['StateHoliday_cat'] = df_test_store['StateHoliday_cat'].astype('float')
df_test_store['Assortment_cat'] = df_test_store['Assortment_cat'].astype('float')
df_test_store['StoreType_cat'] = df_test_store['StoreType_cat'].astype('float')
df_test_store['PromoInterval_cat'] = df_test_store['PromoInterval_cat'].astype('float')


df_test_store["is_holiday_state"] = df_test_store['StateHoliday'].map({"0": 0, "a": 1, "b": 1, "c": 1})
df_test_store['dayofyear_cos'] = np.cos(2 * np.pi * df_test_store.Date.dt.dayofyear/max(df_test_store.Date.dt.dayofyear))
df_test_store['dayofyear_sin'] = np.sin(2 * np.pi * df_test_store.Date.dt.dayofyear/max(df_test_store.Date.dt.dayofyear))
df_test_store['weekofyear_cos'] = np.cos(2 * np.pi *df_test_store.Date.dt.weekofyear/max(df_test_store.Date.dt.weekofyear))
df_test_store['weekofyear_sin'] = np.sin(2 * np.pi *df_test_store.Date.dt.weekofyear/max(df_test_store.Date.dt.weekofyear))


df_test_store['Year']= df_test_store.Date.dt.year
df_test_store['Year_cos'] = np.cos(2 * np.pi *df_test_store.Date.dt.year/max(df_test_store.Date.dt.year))
df_test_store['Year_sin'] = np.sin(2 * np.pi *df_test_store.Date.dt.year/max(df_test_store.Date.dt.year))

df_test_store['Day'] = df_test_store.Date.dt.day
df_test_store['Month'] = df_test_store.Date.dt.month
df_test_store['Day_cos'] = np.cos(2 * np.pi *df_test_store.Date.dt.day/max(df_test_store.Date.dt.day))
df_test_store['Day_sin'] = np.sin(2 * np.pi *df_test_store.Date.dt.day/max(df_test_store.Date.dt.day))
df_test_store['month_cos'] = np.cos(2 * np.pi *df_test_store.Date.dt.month/max(df_test_store.Date.dt.month))
df_test_store['month_sin'] = np.sin(2 * np.pi *df_test_store.Date.dt.month/max(df_test_store.Date.dt.month))

df_test_store['CompetitionOpenSince'] = np.where((df_test_store['CompetitionOpenSinceMonth']==0) & (df_test_store['CompetitionOpenSinceYear']==0) , 0,(df_test_store.Month - df_test_store.CompetitionOpenSinceMonth) + 
                                       (12 * (df_test_store.Year - df_test_store.CompetitionOpenSinceYear)) )
df_test_store["is_holiday_state"] = df_test_store['StateHoliday'].map({"0": 0, "a": 1, "b": 1, "c": 1})
df_test_store=pd.get_dummies(df_test_store, columns=["Assortment", "StoreType","PromoInterval"], prefix=["is_Assortment", "is_StoreType","is_PromoInteval"])






























##      train

def rmspe(y, yhat):
    rmspe = np.sqrt(np.mean( (y - yhat)**2 ))
    return rmspe



features = df_train_store.drop(['Customers', 'Sales', 'SalesperCustomer'], axis = 1)
targets = df_train_store.Sales



X_train, X_train_test, y_train, y_train_test = \
    model_selection.train_test_split(features, targets, test_size=0.20, random_state=15)



rfr = RandomForestRegressor(n_estimators=10, 
                             criterion='mse', 
                             max_depth=5, 
                             min_samples_split=2, 
                             min_samples_leaf=1, 
                             min_weight_fraction_leaf=0.0, 
                             max_features='auto', 
                             max_leaf_nodes=None, 
                             min_impurity_decrease=0.0, 
                             min_impurity_split=None, 
                             bootstrap=True, 
                             oob_score=False,
                             n_jobs=8,
                             random_state=31, 
                             verbose=0, 
                             warm_start=False)









rfr.fit(X_train, y_train)

params = {'max_depth':(4,6,8,10,12,14,16,20),
         'n_estimators':(4,8,16,24,48,72,96,128),
         'min_samples_split':(2,4,6,8,10)}

scoring_fnc = metrics.make_scorer(rmspe)

grid = model_selection.RandomizedSearchCV(estimator=rfr,param_distributions=params,cv=10)

grid.fit(X_train, y_train)

grid.best_params_,grid.best_score_


rfr_val=RandomForestRegressor(n_estimators=8, 
                             criterion='mse', 
                             max_depth=20, 
                             min_samples_split=8, 
                             min_samples_leaf=1, 
                             min_weight_fraction_leaf=0.0, 
                             max_features='auto', 
                             max_leaf_nodes=None, 
                             min_impurity_decrease=0.0, 
                             min_impurity_split=None, 
                             bootstrap=True, 
                             oob_score=False,
                             n_jobs=8, #setting n_jobs to 4 makes sure you're using the full potential of the machine you're running the training on
                             random_state=35, 
                             verbose=1, 
                             warm_start=False)
                            
model_RF_test = rfr_val.fit(X_train,y_train)
yhat = model_RF_test.predict(X_train_test)
plt.hist(yhat)
error = rmspe(y_train_test,yhat)
error






importances = rfr_val.feature_importances_
std = np.std([rfr_val.feature_importances_ for tree in rfr_val.estimators_],
             axis=0)
indices = np.argsort(importances)
palette1 = itertools.cycle(sns.color_palette())
# Store the feature ranking
features_ranked=[]
for f in range(X_train.shape[1]):
    features_ranked.append(X_train.columns[indices[f]])
# Plot the feature importances of the forest

plt.figure(figsize=(10,15))
plt.title("Feature importances")
plt.barh(range(X_train.shape[1]), importances[indices],
            color=[next(palette1)], align="center")
plt.yticks(range(X_train.shape[1]), features_ranked)
plt.ylabel('Features')
plt.ylim([-1, X_train.shape[1]])
plt.show()