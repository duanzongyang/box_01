import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *
from sklearn.linear_model import ElasticNet,Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.stats import norm,skew
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import xgboost as xgb
# import lightgbm as lgb
warnings.filterwarnings('ignore')
train = pd.read_csv('1444_TOTAL.csv', usecols=(0, 1, 2, 3, 4, 5, 6, 7))
test = pd.read_csv('1444_true_test.csv', usecols=(0, 1, 2, 3, 4, 5, 6))
print(test.shape)
ytrain = train['F']
xtrain = train.drop(labels=['F'], axis=1, inplace=False)
print(xtrain)
n_folds=5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(xtrain.values)
    rmse = np.sqrt(-cross_val_score(model, xtrain.values, ytrain, scoring="neg_mean_squared_error", cv=kf))
    return rmse
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005,random_state=1))
ENet=make_pipeline(RobustScaler(),ElasticNet(alpha=0.0005,l1_ratio=.9,random_state=3))
KRR=KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=5)
xgb_model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200, reg_alpha=0.4640, reg_lambda=0.8571,
                              subsample=0.5213, silent=1, random_state=7, nthread=-1)
# lgb_model =lgb.LGBMRegressor(objective='regression',num_leaves=1000,learning_rate=0.05,n_estimators=350,reg_alpha=0.9)
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(xgb_model)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = rmsle_cv(lgb_model)
# print("lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)
averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))
score_all = rmsle_cv(averaged_models)
print('Averaged base models score: {:.4f} ({:.4f})\n'.format(score_all.mean(), score_all.std()))

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
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

stacked_averaged_models.fit(xtrain.values, ytrain)
stacked_train_pred = stacked_averaged_models.predict(xtrain.values)
stacked_test_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(new_rmsle(ytrain, stacked_train_pred))
result = pd.DataFrame()
result['F'] = stacked_test_pred
result.to_csv('result123.csv')

