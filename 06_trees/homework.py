
### Preparation
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

df_original = pd.read_csv('car_fuel_efficiency.csv')

df = df_original.fillna(0)

TARGET_COL = 'fuel_efficiency_mpg'
RANDOM_STATE = 1

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=RANDOM_STATE)

y_train = df_train[TARGET_COL].values
y_val = df_val[TARGET_COL].values
y_test = df_test[TARGET_COL].values

df_train.pop(TARGET_COL)
df_val.pop(TARGET_COL)
df_test.pop(TARGET_COL)

dv = DictVectorizer(sparse=True)

dict_train = df_train.to_dict(orient='records')
dict_val = df_val.to_dict(orient='records')

X_train = dv.fit_transform(dict_train)
X_val = dv.transform(dict_val)

### Question 1

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text

dtr = DecisionTreeRegressor(max_depth=1, random_state=RANDOM_STATE)
dtr.fit(X_train, y_train)

print(export_text(dtr, feature_names=list(dv.get_feature_names_out())))
"""
Split feature: vehicle_weight
|--- vehicle_weight <= 3022.11
|   |--- value: [16.88]
|--- vehicle_weight >  3022.11
|   |--- value: [12.94]
"""

### Question 2

from sklearn.ensemble import RandomForestRegressor

rfg = RandomForestRegressor(n_estimators=10, random_state=RANDOM_STATE, n_jobs=-1)
rfg.fit(X_train, y_train)

def rmse(y, y_pred):
    mse = np.mean((y - y_pred) ** 2)
    return np.sqrt(mse)

y_pred = rfg.predict(X_val)

print(rmse(y_val, y_pred)) # 0.4599777557336149

# Closest answer: 0.45


### Question 3

rmse_scores = {}

for n in range(10, 201, 10):
    rfg = RandomForestRegressor(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1)
    rfg.fit(X_train, y_train)
    y_pred = rfg.predict(X_val)

    score = rmse(y_val, y_pred)
    rmse_scores[n] = round(score, 3)

rmse_df = pd.DataFrame({
    'n_estimators': list(rmse_scores.keys()),
    'rmse': list(rmse_scores.values())
})

plt.plot(rmse_df['n_estimators'], rmse_df['rmse']) # 200



### Question 4

scores = []
max_depth_list = [10, 15, 20, 25]

for d in max_depth_list:
    for n in range(10, 201, 10):
        rfg = RandomForestRegressor(
            n_estimators=n, max_depth=d,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        rfg.fit(X_train, y_train)
        y_pred = rfg.predict(X_val)

        rmse_score = rmse(y_val, y_pred)
        scores.append((d, n, rmse_score))
        
cols = ['max_depth', 'n_estimators', 'rmse']

df_rmse = pd.DataFrame(scores, columns = cols)

for d in max_depth_list:
    df_subset = df_rmse[df_rmse.max_depth == d]
    
    plt.plot(df_subset.n_estimators, df_subset.rmse,
             label='max_depth=%d' % d)

plt.legend() # 10


### Question 5

rfg_q5 = RandomForestRegressor(
    n_estimators=10, max_depth=20,
    random_state=RANDOM_STATE, n_jobs=-1
)

rfg_q5.fit(X_train, y_train)


feature_names = dv.get_feature_names_out()
importances = rfg_q5.feature_importances_

fi_df = pd.DataFrame(
    {
     'feature': feature_names,
     'importance': importances
})
fi_df.sort_values(by='importance', inplace=True, ascending=False) # vehicle_weight with 0.959 - Next one is horsepower with 0.01


### Question 6
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(feature_names))
dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(feature_names))


xgb_params_1 = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

watchlist = [(dtrain, 'train'), (dval, 'validation')]

model_1 = xgb.train(
    xgb_params_1,
    dtrain,
    num_boost_round=100,
    verbose_eval=10,
    evals=watchlist
)
# [99]	train-rmse:0.21950	validation-rmse:0.45018

xgb_params_2 = {
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

model_2 = xgb.train(
    xgb_params_2,
    dtrain,
    num_boost_round=100,
    verbose_eval=10,
    evals=watchlist
)
# [99]	train-rmse:0.30419	validation-rmse:0.42623


## Using last round as comparison, best result is with eta = 0.1
