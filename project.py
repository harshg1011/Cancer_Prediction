import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

df = pd.read_csv("cancer.csv")
df.head()
df.columns

df.shape
df.info()

df.isnull().sum()
df.describe()
sns.countplot(x = 'Status', data = df)
plt.title("Patient Status Distribution")
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x='Cancer_Type', data=df)
plt.xticks(rotation=90)
plt.show()

sns.countplot(x='Stage',data=df)
plt.show()

sns.histplot(df['Survival_Months'], bins = 30, kde= True)
plt.show()

df.drop(columns =['Patient_ID'], inplace = True)

df['Diagnosis_Date'] = pd.to_datetime(df['Diagnosis_Date'])

df['Diagnosis_Year'] = df['Diagnosis_Date'].dt.year
df['Diagnosis_Month'] = df['Diagnosis_Date'].dt.month

df.drop(columns=['Diagnosis_Date'], inplace=True)

X = df.drop('Status', axis=1)
y = df['Status']

categorical_cols = X.select_dtypes(include = 'object').columns
numerical_cols = X.select_dtypes(include =['int64','float64']).columns

print("Categorical:")
print(categorical_cols)


print("\nNumerical:")
print(numerical_cols)

preprocessor = ColumnTransformer(
    transformers = [
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown = 'ignore'),
         categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42, stratify = y)

pipeline = Pipeline([
    ('preprocessor',preprocessor),
    ('model',LogisticRegression(
        random_state = 42
    ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy : ", round(accuracy * 100,2), "%")

joblib.dump(pipeline, "model/cancer_model.pkl")


print("Model Saved Successfully")