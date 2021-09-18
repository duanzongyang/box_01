import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# print(df_train.columns)
# print(df_train['SalePrice'].describe())
# sns.distplot(df_train['SalePrice'])
# plt.show()
# print('skewness:%f'%df_train['SalePrice'].skew())
# print('kurtness:%f'%df_train['SalePrice'].kurt())
# plt.scatter(x=df_train['OverallQual'], y=df_train['SalePrice'])
# plt.show()
# f,ax = plt.subplots(figsize=(8,6))
# fig = sns.boxplot(x=df_train['OverallQual'], y=df_train['SalePrice'])
# fig.axis(ymin=0, ymax=800000)
# plt.show()
# f,ax = plt.subplots(figsize=(8,6))
# fig = sns.boxplot(x=df_train['YearBuilt'], y=df_train['SalePrice'])
# fig.axis(ymin=0, ymax=800000)
# plt.xticks(rotation=90)
# plt.show()
# corrmat = df_train.corr()
# f,ax = plt.subplots(figsize=(12,9))
# sns.heatmap(corrmat, square=True, cmap='YlGnBu')
# plt.show()
# plt.scatter(x = df_train['GrLivArea'], y=df_train['SalePrice'])
# plt.show()
# corrmat = df_train.corr()
# print(corrmat)
# sns.set()
# cols=['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
# sns.pairplot(df_train[cols], size=2.5)
# plt.show()
# total = df_train.isnull().sum().sort_values(ascending=False)
# precent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
# missing_data = pd.concat([total, precent], axis=1, keys=['Total','Percent'])
# print(missing_data.head(20))
from scipy.stats import norm,skew
from sklearn.preprocessing import LabelEncoder
from scipy import stats
# print(train.shape)
# print(test.shape)
train_ID = train['Id']
test_ID = test['Id']
train.drop('Id',axis=1, inplace=True)
test.drop('Id',axis=1, inplace=True)
# print(train.shape)
# print(test.shape)
# plt.scatter(x=train['GrLivArea'], y=train['SalePrice'])
# plt.xlabel('GrLivArea')
# plt.ylabel('SalePrice')
# plt.show()
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)
# plt.scatter(x=train['GrLivArea'], y=train['SalePrice'])
# plt.xlabel('GrLivArea')
# plt.ylabel('SalePrice')
# plt.show()
# sns.distplot(train['SalePrice'], fit=norm)
# plt.show()
# (mu, sigma) = norm.fit(train['SalePrice'])
# print('\n mu = {:.2f} and sigma={:.2f}\n'.format(mu, sigma))
# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.show()
train['SalePrice'] = np.log(train['SalePrice'])
# sns.distplot(train['SalePrice'],fit=norm)
# plt.show()
# (mu, sigma) = norm.fit(train['SalePrice'])
# print('mu={:.2f},sigma = {:.2f}'.format(mu, sigma))
# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.show()
ntrain = train.shape[0]
ntest = test.shape[0]
ytrain = train['SalePrice']
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
# print('all_data size is :{}'.format(all_data.shape))
all_data_na_rate = all_data.isnull().sum()/len(all_data)
all_data_na_rate = all_data_na_rate.drop(all_data_na_rate[all_data_na_rate == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing_rate':all_data_na_rate})
# print(missing_data)
# f,ax= plt.subplots(figsize=(15,12))
# plt.xticks(rotation='90')
# sns.barplot(x=all_data_na_rate.index, y = all_data_na_rate)
# plt.xlabel('Features')
# plt.ylabel('Percent of missing values', fontsize=15)
# plt.title('Percent missing data by feature', fontsize=15)
# plt.show()
# print(all_data['PoolQC'][:5])
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
# print(all_data['PoolQC'][:5])
# print(all_data['MiscFeature'][:10])
all_data['MiscFeature']=all_data['MiscFeature'].fillna('None')
all_data['Alley']=all_data['Alley'].fillna('None')
#栅栏
all_data['Fence']=all_data['Fence'].fillna('None')
#壁炉
all_data['FireplaceQu']=all_data['FireplaceQu'].fillna('None')
all_data['LotFrontage']=all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x:x.fillna(x.median()))
#车库的一系列特征
for col in ('GarageFinish','GarageQual','GarageCond','GarageType'):
    all_data[col]=all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
#地下室的一系列特征
for col in ('BsmtFullBath','BsmtUnfSF','TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','BsmtHalfBath'):
    all_data[col]=all_data[col].fillna(0)
for col in ('BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1'):
    all_data[col]=all_data[col].fillna('None')
#砌体
all_data['MasVnrType']=all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea']=all_data['MasVnrArea'].fillna(0)
all_data['MSZoning'].mode()
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
#家庭功能评定  对于Functional，数据描述里说明，其NA值代表Typ
all_data['Functional']=all_data['Functional'].fillna('Typ')
#电力系统
all_data['Electrical']=all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
#厨房品质
all_data['KitchenQual']=all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
#外部
all_data['Exterior1st']=all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd']=all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
#销售类型
all_data['SaleType']=all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
#建筑类型
all_data['MSSubClass']=all_data['MSSubClass'].fillna('None')

all_data=all_data.drop(['Utilities'],axis=1)
all_data_na_1=(all_data.isnull().sum()/len(all_data))*100
all_data_na_1=all_data_na_1.drop(all_data_na_1[all_data_na_1==0].index).sort_values(ascending=False)
missing_data_1=pd.DataFrame({'Missing Ratio':all_data_na_1})
# print(missing_data_1)
all_data['MSSubClass']=all_data['MSSubClass'].apply(str)
all_data['OverallCond']=all_data['OverallCond'].astype(str)
all_data['YrSold']=all_data['YrSold'].astype(str)
all_data['MoSold']=all_data['MoSold'].astype(str)
cols=('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',\
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',\
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',\
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',\
        'YrSold', 'MoSold')
for c in cols:
    labl = LabelEncoder()
    labl.fit(list(all_data[c].values))
    all_data[c] = labl.transform(list(all_data[c].values))
# print('Shape all_data: {}'.format(all_data.shape))
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
umeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = all_data[umeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew': skewed_feats})
# print(skewness.head(10))
skewness = skewness[abs(skewness) > 0.75]
# print('there are {} skewed numerical features to Box Cox transform'.format(skewness.shape[0]))
from scipy.special import boxcox1p
skewed_feats_index = skewness.index
lam = 0.15
for feat in skewed_feats_index:
    all_data[feat] = boxcox1p(all_data[feat], lam)
all_data = pd.get_dummies(all_data)
# print(all_data.shape)
train = all_data[:ntrain]
test = all_data[ntrain:]
from sklearn.linear_model import ElasticNet,Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
# import xgboost as xgb
# import lightgbm as lgb

n_folds=5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, ytrain, scoring="neg_mean_squared_error", cv=kf))
    return rmse

lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005,random_state=1))
ENet=make_pipeline(RobustScaler(),ElasticNet(alpha=0.0005,l1_ratio=.9,random_state=3))
KRR=KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=5)
# xgb_model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3,
#                              min_child_weight=1.7817, n_estimators=2200, reg_alpha=0.4640, reg_lambda=0.8571,
#                              subsample=0.5213, silent=1, random_state=7, nthread=-1)
# lgb_model =lgb.LGBMRegressor(objective='regression',num_leaves=1000,learning_rate=0.05,n_estimators=350,reg_alpha=0.9)
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = rmsle_cv(xgb_model)
# print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = rmsle_cv(lgb_model)
# print("lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        # train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self
    # we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)
# averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))
# score_all = rmsle_cv(averaged_models)
# print('Averaged base models score: {:.4f} ({:.4f})\n'.format(score_all.mean(), score_all.std()))
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR), meta_model = lasso)
score_all_stacked = rmsle_cv(stacked_averaged_models)
print('Stacking Averaged base models score: {:.4f} ({:.4f})\n'.format(score_all_stacked.mean(), score_all_stacked.std()))

def new_rmsle(y,y_predict):
    return np.sqrt(mean_squared_error(y, y_predict))

stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_test_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))
xgb_model.fit(train, y_train)
xgb_train_pred = xgb_model.predict(train)
xgb_test_pred = np.expm1(xgb_model.predict(test))
print(rmsle(y_train, xgb_train_pred))
lgb_model.fit(train, y_train)
lgb_train_pred = lgb_model.predict(train)
lgb_pred = np.expm1(lgb_model.predict(test.values))
print(rmsle(y_train, lgb_train_pred))
print('RMSLE score on train data all models:')
print(rmsle(y_train, stacked_train_pred * 0.7 + xgb_train_pred * 0.15 +lgb_train_pred * 0.15 ))
# Ensemble prediction 集成预测
ensemble_result = stacked_test_pred * 0.80 + xgb_test_pred * 0.10 + lgb_test_pred *0.10
result = pd.DataFrame()
result['Id'] = test_ID
result['F'] = ensemble_result
result.to_csv('result.csv')

















