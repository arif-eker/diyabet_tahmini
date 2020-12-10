# Kütüphaneler ekleniyor.

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import scripts.models as models
import scripts.helper_functions as hlp

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv("data/diabetes.csv")

df.head()
df.tail()

df.info()

df.columns

df.shape

for col in df.columns:
    print(col, " : ", df[col].nunique(), " eşsiz sınıfa sahiptir.")

df.isnull().sum()

# 0 olan değerlere NaN atayalım
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols:
    df[col].replace(0, np.NaN, inplace=True)

for col in df.columns:
    if col == "Outcome":
        pass

    else:
        df.loc[(df["Outcome"] == 0) & (df[col].isnull()), col] = df[df["Outcome"] == 0][col].median()
        df.loc[(df["Outcome"] == 1) & (df[col].isnull()), col] = df[df["Outcome"] == 1][col].median()

# Yaş değişkenlerinden kategorik değişken üretelim :
bins = [20, 30, 45, 55, 100]
names = ['Young', 'Adult', 'Mature', 'Old']
df["Age_Range"] = pd.cut(df['Age'], bins, labels=names)

df["Age_Range"] = df["Age_Range"].astype("object")

num_cols = [col for col in df.columns if df[col].dtypes != "O"
            and col != "Outcome"]

hlp.has_outliers(df, num_cols)

hlp.replace_with_thresholds(df, num_cols)

# Yeni değişkenler türetelim.
df['New_Glucose_Class'] = pd.cut(x=df['Glucose'], bins=[0, 139, 200], labels=["Normal", "Prediabetes"])

df['New_BMI_Range'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],
                             labels=["Underweight", "Healty", "Overweight", "Obese"])

df['New_BloodPressure'] = pd.cut(x=df['BloodPressure'], bins=[0, 79, 89, 123], labels=["Normal", "HS1", "HS2"])

df['New_SkinThickness'] = df['SkinThickness'].apply(lambda x: 1 if x <= 18.0 else 0)

df["New_BMI*BLOODPRESSURE"] = df['BloodPressure'] * df['BMI']

df["New_Glucose*Insulin"] = df['Glucose'] * df["Insulin"]

df["New_BLOODPRESSURE/Ins*Glu"] = df['BloodPressure'] / (df['Glucose'] * df["Insulin"])

categorical_columns = [col for col in df.columns
                       if len(df[col].unique()) <= 10
                       and col != "Outcome"]

df, new_cols_ohe = hlp.one_hot_encoder(df, categorical_columns)

scale_cols = [col for col in df.columns
              if col not in categorical_columns
              and col not in new_cols_ohe
              and col != "Outcome"]

for col in scale_cols:
    df[col] = hlp.robust_scaler(df[col])

# Modelleme
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

gbm_tuned, lgbm_tuned, rf_tuned, xgb_tuned = models.get_tuned_models(X_train, y_train, 123)

# Modelimizi verimizin %80 i ile eğittik. Verimize train verimizi sorduk öncelikle.
models = [('RF', rf_tuned),
          ('GBM', gbm_tuned),
          ("LightGBM", lgbm_tuned),
          ("XGB", xgb_tuned)]

results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=123)
    cv_results = cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Test hatalarımız.
for name, model in models:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    msg = "%s: (%f)" % (name, acc)
    print(msg)
