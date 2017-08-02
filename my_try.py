from sklearn.datasets import load_boston,load_iris
from sklearn.preprocessing import MinMaxScaler,Imputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn import metrics
import numpy as np

def test_clf(clf):
    alpha_can=np.logspace(-3,2,10)
    models=GridSearchCV(clf,param_grid={'alpha':alpha_can},cv=5)
    if hasattr(clf,'C'):
        C_can=np.logspace(1,3,3)
        gamma_can=np.logspace(-3,0,3)
        models.set_params(param_grid={'C':C_can,'gamma':gamma_can})
    if hasattr(clf,'alpha'):
        alpha_can=np.logspace(-3,2,10)
        models.set_params(param_grid={'alpha':alpha_can})
    if hasattr(clf,'max_depth'):
        depth_can=np.linspace(2,5,4)
        models.set_params(param_grid={'max_depth':depth_can})
    if hasattr(clf,'n_neighbors'):
        n_can=np.linspace(2,4,3).astype('int')
        models.set_params(param_grid={'n_neighbors':n_can})
    models.fit(x_train,y_train)
    predicted=models.predict(x_test)
    acc=metrics.accuracy_score(y_test,predicted)
    name=str(clf).split('(')[0]
    print('最有参数是：',models.best_estimator_)
    print(name+'的准确度是：%f'% acc)
    #return acc

iris=load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7)

imputer=Imputer(missing_values='NaN',strategy='mean',axis=1)
scaler=MinMaxScaler()
pca=PCA(n_components=2)
select=SelectKBest(k=1)
combined_features=FeatureUnion([('pca',pca),('select',select)])

clfs=[SVC(),MultinomialNB(),RandomForestClassifier(n_estimators=200),
     KNeighborsClassifier(),RidgeClassifier()]

for clf in clfs:
    pip=Pipeline([('imputer',imputer),('scaler',scaler),('pca',pca),('select',select),('combined',combined_features),('classifier',test_clf(clf))])
