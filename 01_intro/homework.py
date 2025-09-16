### Homework - Introduction to Machine Learning

## Question 1
import pandas as pd
print(pd.__version__) # 2.3.2


## Question 2
# Dataset downloaded manually
df = pd.read_csv(r'car_fuel_efficiency.csv')

# Records count
print(len(df)) # 9704


## Question 3

# Check columns available
print(df.columns)

# Num fuel types
print(df['fuel_type'].nunique()) # 2


## Question 4
# Check columns with missing values
col_missing_val = df.isnull().sum()
print(col_missing_val)

# Num columns with missing values
sum(col_missing_val != 0) # 4


## Question 5
df_asia =  df.loc[df['origin']=='Asia']
print(df_asia['fuel_efficiency_mpg'].max()) # 23.759122836520497


## Question 6

# Median value
print(df['horsepower'].median()) # 149
# Mode value (most frequent value)
hp_mode = df['horsepower'].mode()[0]
print(hp_mode)   # 152

# Replace NaN
df_treated = df.copy()
df_treated['horsepower'] = df_treated['horsepower'].fillna(hp_mode)

print(df_treated['horsepower'].median()) # 152


## Question 7
df_asia_reduced = df_asia[['vehicle_weight', 'model_year']].head(7)

import numpy as np
X = np.array(df_asia_reduced)

# Transpose
X_t =  X.T

# Matrix-Matrix multiplication
XTX = X_t.dot(X)

XTX_inv = np.linalg.inv(XTX)

y = [1100, 1300, 800, 900, 1000, 1100, 1200]

w = XTX_inv.dot(X_t).dot(y)
print(w.sum()) # 0.5187709081074007