
import pandas as pd

df_full = pd.read_csv(r'car_fuel_efficiency.csv')

df = df_full[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']].copy()

# EDA
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df['fuel_efficiency_mpg'], bins=50) # No long-tail

# Question 1
print(df.isna().sum()) # Horsepower


# Question 2
median_df = df['horsepower'].median()
print(median_df) # 149


####
import numpy as np

size_df = len(df) # 9704

def split_data(df, val_size=0.2, test_size=0.2, seed=42):
    n = len(df)
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)

    n_val = int(n * val_size)
    n_test = int(n * test_size)
    n_train = n - n_val - n_test

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train + n_val]]
    df_test = df.iloc[idx[n_train + n_val:]]

    return df_train, df_val, df_test

# Question 3
def rmse(y, y_pred):
    mse = np.mean((y - y_pred) ** 2)
    return np.sqrt(mse)

def train_linear_regression(X, y, r=0):
    X = np.array(X)
    y = np.array(y)

    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]

def get_target_data(df, target_col):
    df = df.copy()
    # No need for applying log since it does not have a long tail
    # y = np.log1p(df[target_col].values) 
    y = df[target_col].values
    df = df.drop(columns=[target_col], inplace=False)
    return df, y

def prepare_X(df, fill_value):
    df_filled = df.copy()
    df_filled = df_filled.fillna(fill_value)
    X = df_filled.values
    return X

def validating_model(df_train, df_val, y_train, y_val, fill_val=0, r=0):
    X_train = prepare_X(df_train, fill_value=fill_val)
    w0, w = train_linear_regression(X_train, y_train, r=r)

    X_val = prepare_X(df_val, fill_value=fill_val)
    y_pred = w0 + X_val.dot(w)
    rmse_val = rmse(y_val, y_pred)
    return rmse_val

df_train, df_val, df_test = split_data(df, seed=42)

df_train, y_train = get_target_data(df_train, 'fuel_efficiency_mpg')
df_val, y_val     = get_target_data(df_val,   'fuel_efficiency_mpg')
df_test, y_test   = get_target_data(df_test,  'fuel_efficiency_mpg')

print('With 0:', round(validating_model(df_train, df_val, y_train, y_val, fill_val=0), 2)) # 0.52
print('With Mean:', round(validating_model(df_train, df_val, y_train, y_val, fill_val=df_train['horsepower'].mean()), 2)) # 0.46


# Question 4

reg_values = [0, 0.01, 0.1, 1, 5, 10, 100]
for r in reg_values:
    result = round(validating_model(df_train, df_val, y_train, y_val, fill_val=0, r=r), 2)
    print(f'With {r}: {result}') # All return 0.52


# Question 5
seed_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
scores = []
for seed in seed_values:

    df_train_s, df_val_s, df_test_s = split_data(df, seed=seed)
    
    df_train_s, y_train_s = get_target_data(df_train_s, 'fuel_efficiency_mpg')
    df_val_s, y_val_s     = get_target_data(df_val_s,   'fuel_efficiency_mpg')
    df_test_s, y_test_s   = get_target_data(df_test_s,  'fuel_efficiency_mpg')
    
    rmse_val = validating_model(df_train_s, df_val_s, y_train_s, y_val_s, fill_val=0, r=0)
    scores.append(rmse_val)
    print(f'With seed {seed}: {rmse_val}') 

print('Std: ', round(np.std(scores), 3)) # 0.007


# Question 6


df_train, df_val, df_test = split_data(df, seed=9)
df_train, y_train = get_target_data(df_train, 'fuel_efficiency_mpg')
df_val, y_val     = get_target_data(df_val,   'fuel_efficiency_mpg')
df_test, y_test   = get_target_data(df_test,  'fuel_efficiency_mpg')

df_full_train = pd.concat([df_train, df_val])
df_full_train = df_full_train.reset_index(drop=True)

y_full_train = np.concatenate([y_train, y_val])

rmse_val = validating_model(df_full_train, df_test, y_full_train, y_test, fill_val=0, r=0.001)
print(rmse_val)
