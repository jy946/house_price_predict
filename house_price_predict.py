import numpy as np
import pandas as pd
import os
from scipy.stats import*
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from scipy.special import boxcox1p
from itertools import product, chain
from sklearn.preprocessing import MinMaxScaler

#【1】读取数据
#import data
f = 'E:\HousePrice'
os.chdir(f)
train_df = pd.read_csv('train.csv', index_col=0)
test_df = pd.read_csv('test.csv', index_col=0)
#print(tr_df_df.head())
#print(test_df.head())
#print(tr_df_df.shape)
#print(test_df.shape)
#【2】合并数据(先把tr_df_df中的label取出来)
#y_tr_df = np.log1p(tr_df_df.pop("SalePrice"))
#print(y_tr_df.shape)
all_df = pd.concat((train_df.loc[:,'MSSubClass':'SaleCondition'], test_df.loc[:,'MSSubClass':'SaleCondition']), axis=0,ignore_index=True)
#print(all_df.shape)
#(2919, 79)

##查看缺失值

total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(missing_data.head(20))

#numeric_feats = all_df.dtypes[all_df.dtypes != "object"].index
#print(all_df.dtypes[all_df.dtypes == "object"].index)
#skewed_feats = all_df[numeric_feats].apply(lambda x: skew(x.dropna()))
#print(skewed_feats.sort_values(ascending=True))
#skewed_feats=skewed_feats[skewed_feats>0.5].index
all_df=all_df.drop(['FullBath'],axis=1)
##pool
'''
poolqcna = all_df[(all_df['PoolQC'].isnull()) & (all_df['PoolArea'] != 0)][['PoolQC', 'PoolArea']]
areamean = all_df.groupby('PoolQC')['PoolArea'].mean()
for i in poolqcna.index:
    v = all_df.loc[i, ['PoolArea']].values
    all_df.loc[i, ['PoolQC']] = np.abs(v - areamean).astype('float64').idxmin()

#all_df['PoolQC'] = all_df["PoolQC"].fillna("None")
all_df['PoolArea'] = all_df["PoolArea"].fillna(0)
print(all_df['PoolArea'].head(10))
print(all_df['PoolQC'].head(199))
'''
##填充缺失值
#print(all_df['PoolArea'].isnull().any(axis=0))
all_df=all_df.drop(missing_data[missing_data['Percent']>.4].index, axis=1)
all_df[['GarageCond', 'GarageFinish', 'GarageQual', 'GarageType']] = all_df[
    ['GarageCond', 'GarageFinish', 'GarageQual', 'GarageType']].fillna('None')
all_df[['GarageCars', 'GarageArea']] = all_df[['GarageCars', 'GarageArea']].fillna(0)
all_df['Electrical'] = all_df['Electrical'].fillna(all_df['Electrical'].mode()[0])


all_df_na = pd.DataFrame(all_df.isnull().sum(), columns={'missingNum'})
all_df_na['dtype'] = all_df.dtypes
a = pd.Series(all_df.columns)
#print(a)
BsmtList = a[a.str.contains('Bsmt')].values
#print(BsmtList)##属性值相同 互相填充
condition = (all_df['BsmtExposure'].isnull()) & (all_df['BsmtCond'].notnull())  # 3个
all_df.ix[(condition), 'BsmtExposure'] = all_df['BsmtExposure'].mode()[0]

condition1 = (all_df['BsmtCond'].isnull()) & (all_df['BsmtExposure'].notnull())  # 3个
all_df.ix[(condition1), 'BsmtCond'] = all_df.ix[(condition1), 'BsmtQual']

condition2 = (all_df['BsmtQual'].isnull()) & (all_df['BsmtExposure'].notnull())  # 2个
all_df.ix[(condition2), 'BsmtQual'] = all_df.ix[(condition2), 'BsmtCond']

# 对于BsmtFinType1和BsmtFinType2
condition3 = (all_df['BsmtFinType1'].notnull()) & (all_df['BsmtFinType2'].isnull())
all_df.ix[condition3, 'BsmtFinType2'] = 'Unf'

allBsmtNa = all_df_na.ix[BsmtList, :]
allBsmtNa_obj = allBsmtNa[allBsmtNa['dtype'] == 'object'].index
allBsmtNa_flo = allBsmtNa[allBsmtNa['dtype'] != 'object'].index
all_df[allBsmtNa_obj] = all_df[allBsmtNa_obj].fillna('None')
all_df[allBsmtNa_flo] = all_df[allBsmtNa_flo].fillna(0)

#####MasVnr
MasVnrM = all_df.groupby('MasVnrType')['MasVnrArea'].median()
mtypena = all_df[(all_df['MasVnrType'].isnull()) & (all_df['MasVnrArea'].notnull())][['MasVnrType', 'MasVnrArea']]
for i in mtypena.index:
    v = all_df.loc[i, ['MasVnrArea']].values
    all_df.loc[i, ['MasVnrType']] = np.abs(v - MasVnrM).astype('float64').idxmin()

all_df['MasVnrType'] = all_df["MasVnrType"].fillna("None")
all_df['MasVnrArea'] = all_df["MasVnrArea"].fillna(0)

##### Ms
#print(all_df[all_df['MSSubClass'].isnull() | all_df['MSZoning'].isnull()][['MSSubClass','MSZoning']])
all_df["MSZoning"] = all_df.groupby("MSSubClass")["MSZoning"].transform(lambda x: x.fillna(x.mode()[0]))
#print(all_df[['Condition1','Condition2']].isnull().sum())
all_df['LotFrontage']=all_df.groupby('Condition1')['LotFrontage'].transform(lambda x: x.fillna(x.mode()[0]))


###其他填充  类别变量可以用众数填充 填充的是类别
all_df['KitchenQual'] = all_df['KitchenQual'].fillna(all_df['KitchenQual'].mode()[0])  # 用众数填充
all_df['Exterior1st'] = all_df['Exterior1st'].fillna(all_df['Exterior1st'].mode()[0])
all_df['Exterior2nd'] = all_df['Exterior2nd'].fillna(all_df['Exterior2nd'].mode()[0])
all_df["Functional"] = all_df["Functional"].fillna(all_df['Functional'].mode()[0])
all_df["SaleType"] = all_df["SaleType"].fillna(all_df['SaleType'].mode()[0])
all_df["Utilities"] = all_df["Utilities"].fillna(all_df['Utilities'].mode()[0])
##除garageblt已补充完缺失值
#print(all_df.isnull().sum()[all_df.isnull().sum()>0])
# 删除离群点
outliers_id1 = train_df[(train_df.GrLivArea>4000) & (train_df.SalePrice<200000)].index
outliers_id2 = train_df[(train_df.TotalBsmtSF>5000) & (train_df.SalePrice<200000)].index
#print(outliers_id1,outliers_id2)
#print(all_df.shape) (2919,73)
all_df=all_df.drop(outliers_id1).reset_index(drop=True)##outliers_id1有两个点
#y_train=train_df.SalePrice.drop(outliers_id1)
y_train=train_df.loc[:,'SalePrice'].reset_index(drop=True).drop(outliers_id1)##索引从数字0开始，否则索引是ID 从1开始
y_train=y_train.reset_index(drop=True)
# all_df 0-2916 (2917,73)
# y_train 0-1457  (1458,)

#print(pc_fc_tr['RoofStyle'].unique())
#print(pc_fc_all['RoofStyle'].unique())
#print(pc_fc_all.shape) #(6列)

#print(pc_fc_all['RoofMatl'].value_counts())
####year
'''
year_map = pd.concat(pd.Series('YearGroup' + str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))
# 将年份对应映射
#print(year_map)
all_df.GarageYrBlt = all_df.GarageYrBlt.map(year_map)
#print(all_df.GarageYrBlt)
all_df['GarageYrBlt']= all_df['GarageYrBlt'].fillna('None')# 必须 离散化之后再对应映射
'''

#print(all_df['MoSold'].head(20))

tr_df=all_df[:y_train.shape[0]]
##print(tr_df.shape)(1458, 72)
##print(y_train.shape)(1458,)

#print(all_df.isnull().sum()[all_df.isnull().sum()>0])
columns_for_pc = ['Exterior1st', 'Exterior2nd', 'RoofStyle', 'Condition1', 'Condition2', 'BldgType']
pc_fc_tr=tr_df.loc[:,columns_for_pc]
#print(pc_fc_tr.head(5))
#print(pc_fc_tr.shape)#(1458, 6)
#print(pc_fc_tr)#(1-1460)
pc_fc_all=all_df.loc[:,columns_for_pc]
#print(tr_df.loc[:, columns_for_pc].info()) #dataframe
#print(pc_fc_all.head(5))


#rint(y_train.isnull().sum())

tr_df=pd.concat([tr_df,y_train],axis=1)
'''
data1=pd.concat([tr_df['MoSold'],tr_df['SalePrice']],axis=1)
data1.plot.scatter(x='MoSold', y='SalePrice', ylim=(0,800000));
plt.show()
'''
#print(tr_df[:3])
#print(tr_df.shape) #(1458, 74)
#print(tr_df.isnull().sum()[tr_df.isnull().sum()>0])
#print(tr_df.isnull().sum())
#print(tr_df.isnull().index)
'''
data1=pd.concat([tr_df['MSSubClass'],tr_df['SalePrice']],axis=1)
data1.plot.scatter(x='MSSubClass',y='SalePrice',ylim=(0,800000))
plt.show()
'''
'''
data2=pd.concat([tr_df['Neighborhood'],tr_df['SalePrice']],axis=1)
#sns.set(style='white', color_codes=True)
sns.swarmplot(x='Neighborhood', y='SalePrice', data=data2)
plt.xticks(rotation=90)
plt.show()
'''
'''
data3=pd.concat([tr_df['GarageFinish'],tr_df['SalePrice']],axis=1)
#sns.set(style='white', color_codes=True)
sns.swarmplot(x='GarageFinish', y='SalePrice', data=data3)
plt.xticks(rotation=90)
plt.show()

data4=pd.concat([tr_df['MasVnrType'],tr_df['SalePrice']],axis=1)
#sns.set(style='white', color_codes=True)
sns.set_style({'grid.color':'red'})
sns.swarmplot(x='MasVnrType', y='SalePrice', data=data4)
plt.xticks(rotation=90)
plt.show()
'''
#print(tr_df[tr_df['MSSubClass']=='60']['SalePrice'].mean())
#print(all_df['MSSubClass'].unique()) 多一条150数据
#print(tr_df['MSSubClass'].unique())
'''
print(all_df['Condition2'].unique()) #多一条150数据
print(tr_df['Condition2'].unique())
print(all_df['MSZoning'].unique())
print(tr_df['MSZoning'].unique())
'''
##以上补全缺失值，删除异常点

Neighborhood_Good = pd.DataFrame(np.zeros((all_df.shape[0],1)), columns=['Neighborhood_Good'])
Neighborhood_Good[all_df.Neighborhood=='NridgHt'] = 1
Neighborhood_Good[all_df.Neighborhood=='Crawfor'] = 1
Neighborhood_Good[all_df.Neighborhood=='StoneBr'] = 1
Neighborhood_Good[all_df.Neighborhood=='NoRidge'] = 1
Neighborhood_Good[all_df.Neighborhood=='Somerst'] = 1
# Neighborhood_Good = (alldata['Neighborhood'].isin(['StoneBr','NoRidge','NridgHt','Timber','Somerst']))*1 #(效果没有上面好)
Neighborhood_Good.name='Neighborhood_Good'# 将该变量加入


#print(all_df.isnull().sum()[all_df.isnull().sum()>0])
##变换特征
##类别用房价均值编码
newer_dwelling = all_df['MSSubClass'].map({20: 1,30: 0,40: 0,45: 0,50: 0,60: 1,70: 0,75: 0,80: 0,85: 0,90: 0,120: 1,150: 0,160: 0,180: 0,190: 0})
newer_dwelling.name= 'newer_dwelling' #修改该series的列名
all_df['MSSubClass']=all_df['MSSubClass'].astype(str)
season = (all_df['MoSold'].isin([5,6,7]))*1
season.name='season'
all_df['MoSold']=all_df['MoSold'].astype(str)
obj_cols=all_df.select_dtypes(exclude=[np.number]).columns
#numeric_cols=all_df.select_dtypes(include=[np.number]).columns
#print(numeric_cols)
#print(obj_cols)
##Index([],dtype='object')
quality_encoded = []
q_encoded=[]
for q in obj_cols:
#def encode(frame, feature):
    ordering = pd.DataFrame()
    ordering['val'] = tr_df[q].unique()
    #print(ordering)
    ordering.index = ordering.val
    ordering['price_mean'] = tr_df[[q, 'SalePrice']].groupby(q).mean()['SalePrice']
    # 上述 groupby()操作可以将某一feature下同一取值的数据整个到一起，结合mean()可以直接得到该特征不同取值的房价均值
    #print(ordering)
    ordering = ordering.sort_values('price_mean') ##默认升序
    ordering['order'] = range(1, ordering.shape[0]+1)
    #print(ordering)
    ordering = ordering['order'].to_dict()
    #print(ordering)
    for attr_v, score in ordering.items():
        # e.g. qualitative[2]: {'Grvl': 1, 'MISSING': 3, 'Pave': 2}
        tr_df.loc[tr_df[q] == attr_v, q+'_E'] = score
        all_df.loc[all_df[q] == attr_v, q+'_E'] = score
    quality_encoded.append(q + '_E')
    q_encoded.append(q + '_E')

# 由于qualitative集合中包含了非数值型变量和伪数值型变量（多为评分、等级等，其取值为1,2,3,4等等）两类
# 因此只需要对非数值型变量进行encode()处理。
# 如果采用One-Hot编码，则整个qualitative的特征都要进行pd,get_dummies()处理

#print(tr_df.head(3))
tr_df.drop(obj_cols, axis=1, inplace=True) # 离散变量已经有了编码后的新变量，因此删去原变量
all_df.drop(obj_cols, axis=1, inplace=True)
#print(all_df.shape)
#(2917,72)
print(q_encoded)
##[ 'MSZoning_E', 'Street_E', 'LotShape_E', 'LandContour_E', 'Utilities_E',
# 'LotConfig_E', 'LandSlope_E', 'Neighborhood_E', 'Condition1_E', 'Condition2_E',
# 'BldgType_E', 'HouseStyle_E', 'RoofStyle_E', 'RoofMatl_E', 'Exterior1st_E', 'Exterior2nd_E', 'MasVnrType_E',
# 'ExterQual_E', 'ExterCond_E', 'Foundation_E', 'BsmtQual_E', 'BsmtCond_E','BsmtExposure_E',
# 'BsmtFinType1_E', 'BsmtFinType2_E', 'Heating_E', 'HeatingQC_E', 'CentralAir_E', 'Electrical_E',
# 'KitchenQual_E', 'Functional_E', 'GarageType_E', 'GarageFinish_E', 'GarageQual_E', 'GarageCond_E',
# 'PavedDrive_E', 'SaleType_E', 'SaleCondition_E']


##以上除Neighborhood 外all_df与tr_df特征一致，以下变换all_df. 以上补全缺失值，删除异常点，将类别转换成数值型特征
#season=all_df['MoSold'].map({1:0, 3:0, 10:0, 2:1, 12:1, 4:2, 8:2, 9:2, 11:2, 7:3, 6:3, 5:3})
'''
season = (all_df['MoSold'].isin([5,6,7]))*1
season.name='season'
all_df['MoSold']=all_df['MoSold'].astype(str)
'''

'''
newer_dwelling = all_df['MSSubClass'].map({20: 1,30: 0,40: 0,45: 0,50: 0,60: 1,70: 0,75: 0,80: 0,85: 0,90: 0,120: 1,150: 0,160: 0,180: 0,190: 0})
newer_dwelling.name= 'newer_dwelling' #修改该series的列名
all_df['MSSubClass']=all_df['MSSubClass'].astype(str)
'''
##对质量拆分新特征
# 处理OverallQual：将该属性分成两个子属性，以5为分界线，大于5及小于5的再分别以序列
overall_poor_qu = all_df.OverallQual.copy()  # Series类型
overall_poor_qu = 5 - overall_poor_qu
overall_poor_qu[overall_poor_qu < 0] = 0
overall_poor_qu.name = 'overall_poor_qu'
overall_good_qu = all_df.OverallQual.copy()
overall_good_qu = overall_good_qu - 5
overall_good_qu[overall_good_qu < 0] = 0
overall_good_qu.name = 'overall_good_qu'

# 处理OverallCond ：将该属性分成两个子属性，以5为分界线，大于5及小于5的再分别以序列
overall_poor_cond = all_df.OverallCond.copy()  # Series类型
overall_poor_cond = 5 - overall_poor_cond
overall_poor_cond[overall_poor_cond < 0] = 0
overall_poor_cond.name = 'overall_poor_cond'
overall_good_cond = all_df.OverallCond.copy()
overall_good_cond = overall_good_cond - 5
overall_good_cond[overall_good_cond < 0] = 0
overall_good_cond.name = 'overall_good_cond'

# 处理ExterQual：将该属性分成两个子属性，以3为分界线，大于3及小于3的再分别以序列
exter_poor_qu = all_df.ExterQual_E.copy()
exter_poor_qu[exter_poor_qu < 3] = 1
exter_poor_qu[exter_poor_qu >= 3] = 0
exter_poor_qu.name = 'exter_poor_qu'
exter_good_qu = all_df.ExterQual_E.copy()
exter_good_qu[exter_good_qu <= 3] = 0
exter_good_qu[exter_good_qu > 3] = 1
exter_good_qu.name = 'exter_good_qu'

# 处理ExterCond：将该属性分成两个子属性，以3为分界线，大于3及小于3的再分别以序列
exter_poor_cond = all_df.ExterCond_E.copy()
exter_poor_cond[exter_poor_cond < 3] = 1
exter_poor_cond[exter_poor_cond >= 3] = 0
exter_poor_cond.name = 'exter_poor_cond'
exter_good_cond = all_df.ExterCond_E.copy()
exter_good_cond[exter_good_cond <= 3] = 0
exter_good_cond[exter_good_cond > 3] = 1
exter_good_cond.name = 'exter_good_cond'

# 处理BsmtCond：将该属性分成两个子属性，以3为分界线，大于3及小于3的再分别以序列
bsmt_poor_cond = all_df.BsmtCond_E.copy()
bsmt_poor_cond[bsmt_poor_cond < 3] = 1
bsmt_poor_cond[bsmt_poor_cond >= 3] = 0
bsmt_poor_cond.name = 'bsmt_poor_cond'
bsmt_good_cond = all_df.BsmtCond_E.copy()
bsmt_good_cond[bsmt_good_cond <= 3] = 0
bsmt_good_cond[bsmt_good_cond > 3] = 1
bsmt_good_cond.name = 'bsmt_good_cond'

# 处理GarageQual：将该属性分成两个子属性，以3为分界线，大于3及小于3的再分别以序列
garage_poor_qu = all_df.GarageQual_E.copy()
garage_poor_qu[garage_poor_qu < 3] = 1
garage_poor_qu[garage_poor_qu >= 3] = 0
garage_poor_qu.name = 'garage_poor_qu'
garage_good_qu = all_df.GarageQual_E.copy()
garage_good_qu[garage_good_qu <= 3] = 0
garage_good_qu[garage_good_qu > 3] = 1
garage_good_qu.name = 'garage_good_qu'

# 处理GarageCond：将该属性分成两个子属性，以3为分界线，大于3及小于3的再分别以序列
garage_poor_cond = all_df.GarageCond_E.copy()
garage_poor_cond[garage_poor_cond < 3] = 1
garage_poor_cond[garage_poor_cond >= 3] = 0
garage_poor_cond.name = 'garage_poor_cond'
garage_good_cond = all_df.GarageCond_E.copy()
garage_good_cond[garage_good_cond <= 3] = 0
garage_good_cond[garage_good_cond > 3] = 1
garage_good_cond.name = 'garage_good_cond'

# 处理KitchenQual：将该属性分成两个子属性，以3为分界线，大于3及小于3的再分别以序列
kitchen_poor_qu = all_df.KitchenQual_E.copy()
kitchen_poor_qu[kitchen_poor_qu < 3] = 1
kitchen_poor_qu[kitchen_poor_qu >= 3] = 0
kitchen_poor_qu.name = 'kitchen_poor_qu'
kitchen_good_qu = all_df.KitchenQual_E.copy()
kitchen_good_qu[kitchen_good_qu <= 3] = 0
kitchen_good_qu[kitchen_good_qu > 3] = 1
kitchen_good_qu.name = 'kitchen_good_qu'
##qu_list二值新属性
qu_list = pd.concat((overall_poor_qu, overall_good_qu, overall_poor_cond, overall_good_cond, exter_poor_qu,
                     exter_good_qu, exter_poor_cond, exter_good_cond, bsmt_poor_cond, bsmt_good_cond, garage_poor_qu,
                     garage_good_qu, garage_poor_cond, garage_good_cond, kitchen_poor_qu, kitchen_good_qu), axis=1)
#print(qu_list.head(3))
#print(qu_list[-3:]) #0-2918
###与时间相关属性
year_map = pd.concat(pd.Series('YearGroup' + str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))
# 将年份对应映射
#print(year_map)
all_df.GarageYrBlt = all_df.GarageYrBlt.map(year_map)
#print(all_df.GarageYrBlt)
all_df['GarageYrBlt']= all_df['GarageYrBlt'].fillna('None')

Xremoded = (all_df['YearBuilt'] != all_df['YearRemodAdd']) * 1  # (@@@@@)
#print(Xremoded)
Xrecentremoded = (all_df['YearRemodAdd'] >= all_df['YrSold']) * 1  # (@@@@@)
XnewHouse = (all_df['YearBuilt'] >= all_df['YrSold']) * 1  # (@@@@@)
XHouseAge = 2010 - all_df['YearBuilt']
#print(XHouseAge)
XTimeSinceSold = 2010 - all_df['YrSold']
XYearSinceRemodel = all_df['YrSold'] - all_df['YearRemodAdd']

Xremoded.name = 'Xremoded'
Xrecentremoded.name = 'Xrecentremoded'
XnewHouse.name = 'XnewHouse'
XTimeSinceSold.name = 'XTimeSinceSold'
XYearSinceRemodel.name = 'XYearSinceRemodel'
XHouseAge.name = 'XHouseAge'

all_df.YearBuilt = all_df.YearBuilt.map(year_map)
all_df.YearRemodAdd = all_df.YearRemodAdd.map(year_map)
all_df=all_df.drop(['YrSold'],axis=1)
year_list = pd.concat((Xremoded, Xrecentremoded, XnewHouse, XHouseAge, XTimeSinceSold, XYearSinceRemodel), axis=1)
#print(year_list.head(3))
#print(all_df.isnull().sum()[all_df.isnull().sum()>0])
#print(all_df[:3])
#print(all_df[-3:]) 0-2916

###构建价格新属性

clf =SVC(C=100, gamma=0.0001, kernel='rbf')
pc = pd.Series(np.zeros(tr_df.shape[0]))
pc[:] = 'pc1'
id1=tr_df[(tr_df.SalePrice>= 150000)&(tr_df.SalePrice< 220000)].index
id2=tr_df[tr_df.SalePrice>= 220000].index
#print(id1,id2)
pc[id1] = 'pc2'
pc[id2] = 'pc3'
#print(pc[-3:]) #0-1457
#print(pc.value_counts())
#print(pc.head(5))
#print(pc.shape) #(1458,)
#print(tr_df['Exterior1st_E'].dtypes) #float64
#print(tr_df.info()) #dtypes: float64(50), int64(24)
#print(type(tr_df['Exterior1st_E'][3])) #<class 'numpy.float64'>
#print(tr_df.loc[:, columns_for_pc].head(10)) #索引从0开始，带列名

#tr_df[columns_for_pc]=tr_df[columns_for_pc].astype(str)

#print(tr_df['Exterior1st_E'].dtypes)#object
#print(tr_df[columns_for_pc].info())#dtypes: object(6) dataframe
#print(tr_df.loc[:, columns_for_pc].info())  #dtypes: object(6) dataframe

#print(pc_fc_tr[:2]) #0-1457
#print(pc_fc_tr[-3:])
#print(pc_fc_tr.info())#datsframe
X_t = pd.get_dummies(pc_fc_tr, sparse=True)
#print(X_t.shape)
#print(tr_df.loc[:, columns_for_pc])  原值
#print(X_t.head(5))

### get_dummies后数据过于稀疏，必须归一化：由于维度太大，如果不采用归一化处理的话，各个点间的距离值将非常大，故模型对于待预测点的预测结果值都判为同一个值。
scale = StandardScaler()
scale_fit = scale.fit(X_t)
x1= scale_fit.transform(X_t)

#print(x1.shape) #(1458,59)
clf.fit(x1, pc)  # 训练

p = tr_df.SalePrice / 100000

'''
price_category = pd.DataFrame(np.zeros((tr_df.shape[0], 1)), columns=['pc'])
X_t = pd.get_dummies(pc_fc_tr, sparse=True)
pc_pred = clf.predict(X_t)
price_category[pc_pred == 'pc2'] = 1
price_category[pc_pred == 'pc3'] = 2
print(price_category['pc'].value_counts())
'''
#print(pc_fc_all[:2])
#print(pc_fc_all[-3:]) #0-2916
#print(all_df.shape)(2917,73)
price_category = pd.DataFrame(np.zeros((all_df.shape[0], 1)), columns=['pc'])
#all_df[columns_for_pc]=all_df[columns_for_pc].astype(str)
#X_t = pd.get_dummies(all_df.loc[:, columns_for_pc], sparse=True)
X_t = pd.get_dummies(pc_fc_all, sparse=True)
#print(X_t.shape)
#print(X_t.head(5))

x2= scale_fit.transform(X_t)
#print(x2)
#print(x2.shape)
pc_pred = clf.predict(x2)  # 预测

price_category[pc_pred == 'pc2'] = 1
price_category[pc_pred == 'pc3'] = 2
#print(price_category[pc_pred == 'pc2'])#Empty DataFrame Columns: [pc] Index: []
print(price_category['pc'].value_counts())
#print(price_category[:2])
#print(price_category[-2:]) #0-2916
price_category.name = 'price_category'

#object_cols=all_df.select_dtypes(exclude=[np.number]).columns
#print(object_cols)
#####以上构建qu_list yearlist price新属性

##比例缩放
numeric_feats = all_df.dtypes[all_df.dtypes != "object"].index
t = all_df[numeric_feats].quantile(.75) # 取四分之三分位
use_75_scater = t[t != 0].index
all_df[use_75_scater] = all_df[use_75_scater]/all_df[use_75_scater].quantile(.75)
#print(all_df[use_75_scater].head(3))

##标准化数值相差太大的数值型特征
t = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
     '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
     'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
# alldata.loc[:, t] = np.log1p(alldata.loc[:, t])
y_train = np.log1p(y_train)  # 对于SalePrice 采用log1p较好---np.expm1(clf1.predict(X_test))

lam = 0.15  # 100 * (1-lam)% confidence
for feat in t:
    all_df[feat] = boxcox1p(all_df[feat], lam)  # 对于其他属性，采用boxcox1p较好
#print(all_df.shape) (2917,72)

#print(all_df.isnull().sum()[all_df.isnull().sum()>0])
object_feats = all_df.dtypes[all_df.dtypes == "object"].index
print(object_feats)
#Index(['MSSubClass', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold'], dtype='object')


#print(all_df.dtypes)
##离散化剩下的yearbuilt等少量类别属性
X = pd.get_dummies(all_df)
#print(X.shape) (2917,87)


X = pd.concat((X,year_list, qu_list, price_category,newer_dwelling,season,Neighborhood_Good),axis=1)
#print(X[:3])
#print(X[-3:]) #0-2916
#print(year_list[:3])
#print(year_list[-3:])
#print(year_list.shape) #(2917,6)
#print(qu_list.shape) #(2917,16)
#print(price_category.shape) #(2917,1)
#print(X.shape) #(2917,110)
#print(X.isnull().sum()[X.isnull().sum()>0])  #Series([], dtype: int64)
###以上标准化数据

# chain(iter1, iter2, ..., iterN):
# 给出一组迭代器(iter1, iter2, ..., iterN)，此函数创建一个新迭代器来将所有的迭代器链接起来，
# 返回的迭代器从iter1开始生成项，知道iter1被用完，然后从iter2生成项，这一过程会持续到iterN中所有的项都被用完。
ordinalList = ['ExterQual_E', 'ExterCond_E', 'GarageQual_E', 'GarageCond_E',\
               'KitchenQual_E', 'HeatingQC_E', 'BsmtQual_E','BsmtCond_E']###编码后的质量类别特征
#print(type(qu_list)) #dataframe
#print(qu_list.axes[1]) #Index(['overall_poor_qu',''],dtype='object') 取列名  index类型
#print(qu_list.axes[1].get_values()) #['' '']
#print(type(qu_list.axes[1].get_values()))  #<class 'numpy.ndarray'>
def poly(X):
    areas = ['LotArea', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'BsmtUnfSF']  # 5个 标准化后的数值特征(本身就是数值)
    t = chain(qu_list.axes[1].get_values(),year_list.axes[1].get_values(),ordinalList)  #,'Neighborhood_Good','SaleCondition_PriceDown'
    for a, t in product(areas, t):
        x = X.loc[:, [a, t]].prod(1)#.dropna(axis=1) # 返回各维数组的乘积
        x.name = a + '_' + t
        yield x
###以上迭代器构建新特征

##拼接数据
XP = pd.concat(poly(X), axis=1) # (2917, 150)
X = pd.concat((X, XP), axis=1) # (2917, 289)
#print(XP.shape)
#print(X.shape)
#print(X.isnull().sum()[X.isnull().sum()>0])
X_train = X[:y_train.shape[0]]
X_test = X[y_train.shape[0]:]

#print(X_train.shape)  #(1458,276)
#print(y_train.shape)  #(1458,)

#print(X_test[:3])
#print(X_test[-3:])
train_now=pd.concat([X_train,y_train],axis=1)
test_now=X_test
train_now.to_csv('train_afterchange.csv')
test_now.to_csv('test_afterchange.csv')


'''
from scipy.stats import boxcox_normmax
from scipy.special import boxcox1p
lambda_2=boxcox_normmax(trains.SalePrice+1)
print(lambda_2)
trains.SalePrice=boxcox1p(trains.SalePrice,lambda_2)
'''
'''
year_map = pd.concat(pd.Series('YearGroup' + str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))
# 将年份对应映射
all_df.YearBuilt = all_df.YearBuilt.map(year_map)
all_df.YearRemodAdd = all_df.YearRemodAdd.map(year_map)
'''

'''
def encode(frame, feature):

##    对所有类型变量，依照各个类型变量的不同取值对应的样本集内房价的均值，按照房价均值高低
##    对此变量的当前取值确定其相对数值1,2,3,4等等，相当于对类型变量赋值使其成为连续变量。
##    此方法采用了与One-Hot编码不同的方法来处理离散数据，值得学习
##    注意：此函数会直接在原frame的DataFrame内创建新的一列来存放feature编码后的值。

    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['price_mean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    # 上述 groupby()操作可以将某一feature下同一取值的数据整个到一起，结合mean()可以直接得到该特征不同取值的房价均值
    ordering = ordering.sort_values('price_mean')
    ordering['order'] = range(1, ordering.shape[0]+1)
    ordering = ordering['order'].to_dict()
    for attr_v, score in ordering.items():
        # e.g. qualitative[2]: {'Grvl': 1, 'MISSING': 3, 'Pave': 2}
        frame.loc[frame[feature] == attr_v, feature+'_E'] = score

quality_encoded = []
# 由于qualitative集合中包含了非数值型变量和伪数值型变量（多为评分、等级等，其取值为1,2,3,4等等）两类
# 因此只需要对非数值型变量进行encode()处理。
# 如果采用One-Hot编码，则整个qualitative的特征都要进行pd,get_dummies()处理
for q in obj_cols:
    encode(tr_df, q)
    quality_encoded.append(q+'_E')
#print(tr_df.head(3))
tr_df.drop(obj_cols, axis=1, inplace=True) # 离散变量已经有了编码后的新变量，因此删去原变量
'''

#print(tr_df.head(3))
# tr_df.shape = (1460, 80)
#print(quality_encoded, '\n{} qualitative attributes have been encoded.'.format(len(quality_encoded)))

##互相关
###斯皮尔曼等级相关系数
'''
def spearman(frame, features):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['corr'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]
    spr = spr.sort_values('corr')
    plt.figure(figsize=(4, len(features)))
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    sns.barplot(data=spr, y='feature', x='corr', orient='h')
features = list(numeric_cols)+quality_encoded
spearman(tr_df, features)
plt.show()
'''


#plt.figure(1,figsize=(12,9))  # 连续型变量相关图
#corrmat = tr_df[list(numeric_cols)].corr()
#sns.heatmap(corrmat,square=True)
#plt.show()


'''
plt.figure(2,figsize=(12,9))  # 等级型变量相关图（离散型和伪数值型变量均已被概括为等级型变量）
corr = tr_df[quality_encoded+['SalePrice']].corr('spearman')
sns.heatmap(corr,square=True)
plt.show()

plt.figure(3,figsize=(12,9)) # 连续型变量-等级型变量相关图
corr = pd.DataFrame(np.zeros([len(numeric_cols)+1, len(quality_encoded)+1]), 
                    index=list(numeric_cols)+['SalePrice'], columns=list(quality_encoded)+['SalePrice'])
for q1 in list(numeric_cols)+['SalePrice']:
    for q2 in list(quality_encoded)+['SalePrice']:
        corr.loc[q1, q2] = tr_df[q1].corr(tr_df[q2], 'spearman')
sns.heatmap(corr,square=True)
plt.xticks(fontsize=6)
plt.show()
'''
###9个最相关属性之间的相关关系
'''
k=10
cols=corrmat.nlargest(k,'SalePrice')['SalePrice'].index
print(cols)
#print(tr_df[cols].values)
#print(tr_df[cols].values.T)
cm=tr_df[cols].corr()
sns.set(font_scale=0.8)
#print(cm)
hm=sns.heatmap(cm,cbar=True,annot=True,square=True,xticklabels=True,yticklabels=True)
plt.xticks(rotation=45)
plt.show()
'''

'''
k=10
cols_far=corrmat.nsmallest(k,'SalePrice')['SalePrice'].index
#print(tr_df[cols].values)
#print(tr_df[cols].values.T)
cm_far=tr_df[cols_far].corr()
sns.set(font_scale=0.8)
#print(cm)
hm_far=sns.heatmap(cm_far,cbar=True,annot=True,square=True,xticklabels=True,yticklabels=True)
plt.xticks(rotation=45)
plt.show()
'''
'''
sns.set()
sns.pairplot(tr_df[cols],kind='reg')
plt.show()
'''
'''
cols=corrmat.nlargest(20,'SalePrice')['SalePrice'].index
print(cols)
for tf in cols:
    sns.distplot(tr_df[tf], fit=norm)
    plt.show()
'''


