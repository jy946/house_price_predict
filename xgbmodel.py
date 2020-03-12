import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from operator import itemgetter
import itertools

import xgboost as xgb
f='E:\HousePrice'
os.chdir(f)
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

train = pd.read_csv("train_afterchange.csv")
test = pd.read_csv("test_afterchange.csv")
alldata = pd.concat((train.iloc[:,1:-1], test.iloc[:,1:]), ignore_index=True)
#print(train.iloc[:,1:-1].head(3))
#print(test.iloc[:,1:].head(3))
#print(alldata.shape) #(2917,276)
#print(test.index.get_values()) #[   0    1    2 ... 1456 1457 1458]

X_train=train.iloc[:,1:-1]
y=train.iloc[:,-1]
X_test=test.iloc[:,1:]
#print(type(X_train)) #DataFrame

# 定义验证函数
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
'''
###Lassso Model
clf1 = LassoCV(alphas = [1, 0.1, 0.001, 0.0005,0.0003,0.0002, 5e-4])
clf1.fit(X_train, y)
lasso_preds = np.expm1(clf1.predict(X_test)) # exp(x) - 1  <---->log1p(x)==log(1+x)
score1 = rmse_cv(clf1)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score1.mean(), score1.std()))

##Elastic net
clf2 = ElasticNet(alpha=0.0005, l1_ratio=0.9)
clf2.fit(X_train, y)
elas_preds = np.expm1(clf2.predict(X_test))

score2 = rmse_cv(clf2)
print("\nElasticNet score: {:.4f} ({:.4f})\n".format(score2.mean(), score2.std()))
'''
# XGBOOST
clf3 = xgb.XGBRegressor(colsample_bytree=0.4,
                        gamma=0.045,
                        learning_rate=0.07,
                        max_depth=20,
                        min_child_weight=1.5,
                        n_estimators=300,
                        reg_alpha=0.65,
                        reg_lambda=0.45,
                        subsample=0.95,
                        objective ='reg:squarederror')
#print(y)
#print(y.values) [12.24  12.10 ```]
#print(type(y)) #<class 'pandas.core.series.Series'>
#print(type(y.values)) #<class 'numpy.ndarray'>
clf3.fit(X_train, y.values)
xgb_preds = np.expm1(clf3.predict(X_test)) #'numpy.ndarray'
#print(xgb_preds[:3]) #一维数组
score3 = rmse_cv(clf3)
print("\nxgb score: {:.4f} ({:.4f})\n".format(score3.mean(), score3.std()))
#print(type(test.index+1461)) #range.RangeIndex
#print(test.index+1461) #RangeIndex(start=1461, stop=2920, step=1)

#final_result = 0.7*lasso_preds + 0.15*xgb_preds+0.15*elas_preds
solution = pd.DataFrame({"Id":test.index+1461, "SalePrice":xgb_preds}, columns=['Id', 'SalePrice'])
solution.to_csv("Submission.csv", index = False)

