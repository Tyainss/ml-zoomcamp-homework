
import pandas as pd

df_original = pd.read_csv(r'course_lead_scoring.csv')

## Data Preparation
categorical_feats = [
    'lead_source', 'industry', 
    'employment_status', 'location',
]

numerical_feats = [
    'number_of_courses_viewed', 'annual_income', 
    'interaction_count', 'lead_score',
]
feats = categorical_feats + numerical_feats

target_col = 'converted'

df_prep = df_original.copy()
df_prep[categorical_feats] = df_prep[categorical_feats].fillna('NA')
df_prep[numerical_feats] = df_prep[numerical_feats].fillna(0)

df = df_prep.copy()

### Question 1
print(df['industry'].mode()) # retail
# (or alternatively)
print(df['industry'].value_counts())

### Question 2

corr = df[numerical_feats].corr() 
# Looking at the df, it's "interaction_count" and "annual_income" with 0.027

## Split the data
from sklearn.model_selection import train_test_split

seed = 42
# Split 60 / 20 / 20
split_1 = 0.2 / 1
split_2 = 0.2 / (1 - split_1)
df_train_full, df_val = train_test_split(df, test_size=split_1, random_state=seed)
df_train, df_test = train_test_split(df_train_full, test_size=split_2, random_state=seed)

y_train = df_train[target_col].values
y_val = df_val[target_col].values

df_train.pop(target_col)
df_val.pop(target_col)

### Question 3
from sklearn.metrics import mutual_info_score

mi_scores = {}
for c in categorical_feats:
    mi = mutual_info_score(df_train_full[target_col], df_train_full[c])
    mi_scores[c] = round(mi, 2)
    
print(mi_scores)
"""
{'lead_source': 0.03,
 'industry': 0.01,
 'employment_status': 0.01,
 'location': 0.0}
""" # lead_source

# Question 4
from sklearn.feature_extraction import DictVectorizer


train_dict = df_train[feats].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)

# Train model
model.fit(X_train, y_train)

# Get X validation to predict
val_dict = df_val[feats].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:, 1]

def accuracy_from_proba(y_pred, y_val, threshold=0.5):
    decision = (y_pred >= threshold)
    return (decision == y_val).mean()

converted_decision = (y_pred >= 0.5)
# print(round((converted_decision == y_val).mean(), 2)) # 0.73
print(round(accuracy_from_proba(y_pred, y_val), 2))



# Question 5
acc_original = (converted_decision == y_val).mean()

remove_feats = ['industry', 'employment_status', 'lead_score']
acc_dict = {}
diff_dict = {}

for rf in remove_feats:
    feats_rf = [c for c in feats if c != rf]
    t_dict_rf = df_train[feats_rf].to_dict(orient='records')
    v_dict_rf = df_val[feats_rf].to_dict(orient='records')

    dv_rf = DictVectorizer(sparse=False)
    dv_rf.fit(t_dict_rf)
    X_train_rf = dv_rf.transform(t_dict_rf)
    X_val_rf = dv_rf.transform(v_dict_rf)

    model_rf = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model_rf.fit(X_train_rf, y_train)

    y_pred_rf = model_rf.predict_proba(X_val_rf)[:, 1]
    acc_val = accuracy_from_proba(y_pred_rf, y_val)

    acc_dict[rf] = acc_val
    diff_dict[rf] = acc_original - acc_val

print(diff_dict)
"""diff_dict
{'industry': -0.010238907849829282,
 'employment_status': -0.013651877133105783,
 'lead_score': -0.0068259385665528916}
"""

""" act_dict
{'industry': 0.7372013651877133,
 'employment_status': 0.7406143344709898,
 'lead_score': 0.7337883959044369}
"""
# employment_status

# Question 6
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(df_train[feats].to_dict(orient='records'))
X_val = dv.transform(df_val[feats].to_dict(orient='records'))



Cs = [0.01, 0.1, 1, 10, 100]
results = {}

for C in Cs:
    model = LogisticRegression(
        solver='liblinear',
        C=C,
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    p_val = model.predict_proba(X_val)[:, 1]
    acc = accuracy_from_proba(p_val, y_val)
    results[C] = round(acc, 3)

# Show all accuracies and pick the winner
print("Validation accuracies by C:", results)
best_C = max(results, key=results.get)
print("Best C:", best_C, "with acc =", results[best_C])
