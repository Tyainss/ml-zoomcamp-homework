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

from sklearn.model_selection import train_test_split

seed = 1
# Split 60 / 20 / 20
split_1 = 0.2 / 1
split_2 = 0.2 / (1 - split_1)
df_train_full, df_val = train_test_split(df, test_size=split_1, random_state=seed)
df_train, df_test = train_test_split(df_train_full, test_size=split_2, random_state=seed)

y_train = df_train[target_col].values
y_val = df_val[target_col].values

df_train.pop(target_col)
df_val.pop(target_col)

### Question 1
from sklearn.metrics import roc_curve, auc, roc_auc_score


auc_results = {}
for n in numerical_feats:
    y_pred = df_train[n].values
    # fpr, tpr, thresholds = roc_curve(y_train, y_pred)
    # auc_val = auc(fpr, tpr)

    # Or use roc_auc_score directly
    auc_val = roc_auc_score(y_train, y_pred)
    if auc_val < 0.5:
        auc_val = roc_auc_score(y_train, -y_pred)  # invert direction
    auc_results[n] = round(auc_val, 3)

print(auc_results) # number_of_courses_viewed



### Question 2
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

dv = DictVectorizer(sparse=False)

# Convert to list of dicts
train_dict = df_train[feats].to_dict(orient='records')
val_dict = df_val[feats].to_dict(orient='records')

# One-hot encode
X_train = dv.fit_transform(train_dict)
X_val = dv.transform(val_dict)

# Identify numeric feature indices inside the transformed matrix
num_feature_names = [f for f in dv.get_feature_names_out() if any(n in f for n in numerical_feats)]
num_indices = [list(dv.get_feature_names_out()).index(f) for f in num_feature_names]

#################
# With Scaling  #
#################

# Scale only numeric columns
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()

X_train_scaled[:, num_indices] = scaler.fit_transform(X_train[:, num_indices])
X_val_scaled[:, num_indices] = scaler.transform(X_val[:, num_indices])

# Train logistic regression
model_scaled = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model_scaled.fit(X_train_scaled, y_train)

# Evaluate
y_pred_scaled = model_scaled.predict_proba(X_val_scaled)[:, 1]
a_scaled = roc_auc_score(y_val, y_pred_scaled)
print(round(a_scaled, 3)) # 0.92

########################
# With another solver  #
########################

model = LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred_lbfgs = model.predict_proba(X_val)[:, 1]

a = roc_auc_score(y_val, y_pred_lbfgs)
print(round(a, 3)) # 0.92


### Question 3

# Train Logistic Regression unscaled
# model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)

y_pred = y_pred_scaled
# y_pred = y_pred_lbfgs

thresholds = np.linspace(0, 1, 101)

tp = np.zeros_like(thresholds, dtype=int)
fp = np.zeros_like(thresholds, dtype=int)
fn = np.zeros_like(thresholds, dtype=int)
tn = np.zeros_like(thresholds, dtype=int)

actual_pos = (y_val == 1)
actual_neg = (y_val == 0)

for i, t in enumerate(thresholds):
    pred_pos = (y_pred >= t)
    pred_neg = ~pred_pos

    tp[i] = (pred_pos & actual_pos).sum()
    fp[i] = (pred_pos & actual_neg).sum()
    fn[i] = (pred_neg & actual_pos).sum()
    tn[i] = (pred_neg & actual_neg).sum()

# Guard divide-by-zero
precision = np.where((tp + fp) > 0, tp / (tp + fp), np.nan)
recall    = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)

# Find threshold where curves intersect (closest point)
diff = np.abs(precision - recall)
idx  = np.nanargmin(diff)
t_star = thresholds[idx]
print("Approx intersection threshold:", round(t_star, 3))

# Plot
import matplotlib.pyplot as plt
plt.plot(thresholds, precision, label="precision")
plt.plot(thresholds, recall, label="recall")
plt.axvline(thresholds[idx], linestyle="--", alpha=0.5)
plt.legend()
plt.xlabel("threshold")
plt.ylabel("score")
plt.title("Precision vs Recall across thresholds")
plt.show()

# Approx threshold: 0.51
# Answer: 0.545

### Question 4
# Recalculate precision and recall without nan        
precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
recall    = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)

den = precision + recall
f1 = np.where(den > 0, 2 * precision * recall / den, 0.0).astype(float)

best_idx = int(np.nanargmax(f1))
best_thr = thresholds[best_idx]
print("Max F1:", round(float(f1[best_idx]), 3), "at threshold", round(best_thr, 3)) # Max F1: 0.895 at threshold 0.56


### Question 5

from sklearn.model_selection import KFold


df_full = df_train_full.copy()

y_full = df_full[target_col].to_numpy()
X_full = df_full[feats]

kf = KFold(n_splits=5, shuffle=True, random_state=1)
scores = []

for train_idx, val_idx in kf.split(X_full):
    df_tr = X_full.iloc[train_idx]
    df_va = X_full.iloc[val_idx]
    y_tr  = y_full[train_idx]
    y_va  = y_full[val_idx]

    dv = DictVectorizer(sparse=False)
    X_tr = dv.fit_transform(df_tr.to_dict(orient='records'))
    X_va = dv.transform(df_va.to_dict(orient='records'))

    feature_names = list(dv.get_feature_names_out())
    num_cols = [i for i, name in enumerate(feature_names) 
                if any(n in name for n in numerical_feats)]

    # scaler = StandardScaler()
    # X_tr[:, num_cols] = scaler.fit_transform(X_tr[:, num_cols])
    # X_va[:, num_cols] = scaler.transform(X_va[:, num_cols])

    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    model.fit(X_tr, y_tr)
    p_va = model.predict_proba(X_va)[:, 1]
    scores.append(roc_auc_score(y_va, p_va))

mean_auc = float(np.mean(scores))
std_auc  = float(np.std(scores))
print(f"{mean_auc:.3f} +- {std_auc:.3f}") 
# No Scale: 0.822 +- 0.036
# With Scale: 0.921 +- 0.010


### Question 6

c_values = [0.000001, 0.001, 1]

kf = KFold(n_splits=5, shuffle=True, random_state=1)

auc_scores = {}
perf_scores = {}

for C in c_values:
    auc_scores[C] = []
    perf_scores[C] = {}
    for train_idx, val_idx in kf.split(X_full):
        df_tr = X_full.iloc[train_idx]
        df_va = X_full.iloc[val_idx]
        y_tr  = y_full[train_idx]
        y_va  = y_full[val_idx]
    
        dv = DictVectorizer(sparse=False)
        X_tr = dv.fit_transform(df_tr.to_dict(orient='records'))
        X_va = dv.transform(df_va.to_dict(orient='records'))
    
        feature_names = list(dv.get_feature_names_out())
        num_cols = [i for i, name in enumerate(feature_names) 
                    if any(n in name for n in numerical_feats)]
    
        # scaler = StandardScaler()
        # X_tr[:, num_cols] = scaler.fit_transform(X_tr[:, num_cols])
        # X_va[:, num_cols] = scaler.transform(X_va[:, num_cols])
    
        model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        model.fit(X_tr, y_tr)
        p_va = model.predict_proba(X_va)[:, 1]
        auc_scores[C].append(roc_auc_score(y_va, p_va))
        
    perf_scores[C]['mean'] = round(np.mean(auc_scores[C]), 3)
    perf_scores[C]['std'] = round(np.std(auc_scores[C]), 3)

# Answer: 0.001
"""
{1e-06: {'mean': 0.56, 'std': 0.024},
 0.001: {'mean': 0.867, 'std': 0.029},
 1: {'mean': 0.822, 'std': 0.036}}
"""
