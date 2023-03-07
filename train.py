import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from ml_flow import Mlflow
from datetime import datetime

df = pd.read_csv("dataset\dataset.csv", header = 1)
#df.drop([122,123, 167],axis=0, inplace=True)
df = df.reset_index()
df.columns = df.columns.str.strip()

import re
def Remove_Extra_Space(x):
    return (re.sub(' +', ' ', x).strip())
df['Classes'] = df['Classes'].apply(Remove_Extra_Space)
df.drop(['index'],axis=1, inplace=True)
df['Classes'] = df['Classes'].map({'not fire' : 0, 'fire': 1})
X = df[['Temperature', 'RH', 'Rain', 'Ws', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']]
y = df['Classes']

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

x_train,x_test,y_train,y_test = train_test_split(X_scaled,y, test_size= 0.25, random_state = 42)
log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)

import pickle

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
modelpath = timestamp+'modelForPrediction.sav'
scalepath = timestamp+'sandardScalar.sav'
# Writing different model files to file
with open( modelpath, 'wb') as f:
    pickle.dump(log_reg,f)

with open(scalepath, 'wb') as f:
    pickle.dump(scalar,f)

model_path = modelpath


y_pred = log_reg.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
conf_mat = confusion_matrix(y_test,y_pred)
true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]
Precision = true_positive/(true_positive+false_positive)
Recall = true_positive/(true_positive+false_negative)
F1_Score_test = 2*(Recall * Precision) / (Recall + Precision)
auc = roc_auc_score(y_test, y_pred)
print(f" Confusion Matrix : {conf_mat}")
print(f" Accuracy : {accuracy*100}")
print(f" Precision : {Precision*100}")
print(f" Recall : {Recall*100}")
print(f" F1_Score : {F1_Score_test*100}")
print(f" AUC : {auc*100}")

y_pred = log_reg.predict(x_train)

accuracy = accuracy_score(y_train,y_pred)
conf_mat = confusion_matrix(y_train,y_pred)
true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]
Precision = true_positive/(true_positive+false_positive)
Recall = true_positive/(true_positive+false_negative)
F1_Score_train = 2*(Recall * Precision) / (Recall + Precision)
auc = roc_auc_score(y_train, y_pred)
print(f" Confusion Matrix : {conf_mat}")
print(f" Accuracy : {accuracy*100}")
print(f" Precision : {Precision*100}")
print(f" Recall : {Recall*100}")
print(f" F1_Score : {F1_Score_train*100}")
print(f" AUC : {auc*100}")

mlflow_obj = Mlflow(model_path, F1_Score_test, F1_Score_train)
mlflow_obj.create_experiment()