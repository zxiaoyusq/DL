import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.ensemble import RandomForestRegressor

train_data="C:\ml/12.XGBoost\Titanic.train.csv"
test_data="C:\ml/12.XGBoost\Titanic.test.csv"

train_df=pd.read_csv(train_data)
test_df=pd.read_csv(test_data)

train_df.describe(include=np.object)
train_df.describe(include=np.number)

#groupby统计
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#通过筛选条件变更列的值
train_df.loc[train_df['FamilySize'] == 1, 'IsAlone'] = 1
#快速填充空值
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
#根据空值比例进行特征筛选
train_missing = (train_df.isnull().sum() / len(train_df)).sort_values(ascending = False)
train_missing=train_missing.index[train_missing>0.5]
train_df=train_df.drop(columns=train_missing)
#根据多重共线性进行特征筛选
threshold=0.8
corr_matrix = train_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
