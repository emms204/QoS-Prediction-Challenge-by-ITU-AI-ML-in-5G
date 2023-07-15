import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Union, List, Literal

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, Dataset, early_stopping
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from abc import ABC, abstractmethod
import random

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
install("catboost")
from catboost import CatBoostRegressor, Pool

def Load_Data(file_path: str) -> List[pd.DataFrame]:
  train = pd.read_csv(file_path+'/Train.csv')
  test = pd.read_csv(file_path+'/Test.csv')
  sub = pd.read_csv(file_path+'/SampleSubmission.csv')
  variables = pd.read_csv(file_path+'/VariableDefinitions.csv')

  return train, test, sub, variables 

def scatterplot(x,y,hue=None,length=5,width=10):
  fig, ax = plt.subplots(figsize=(width,length))
  sns.scatterplot(x=x,y=y,hue=hue,ax=ax)
  ax.set_xlabel(x.name)
  ax.set_ylabel(y.name)
  ax.set_title(f'{x.name} vs {y.name}')
  plt.show()

def boxplot(df):
  fig, ax = plt.subplots()
  ax.boxplot(df)
  ax.set_xticklabels(list(df.columns))
  ax.set_ylabel('Value')
  ax.set_title(f'Boxplot of {" vs ".join(list(df.columns))}')
  plt.show()

def heatmap(df, length=10, width=5, cmap='rocket'):
  heatmap_df = df.corr()
  fig, ax = plt.subplots(figsize=(length, width))
  sns.heatmap(data=heatmap_df, annot=True, cmap=cmap, ax=ax)
  ax.set_title(f'Heatmap of {" and ".join(list(df.columns))}')
  plt.show()

def violinplot(x, y,hue,split=False,scale=None,length=10, width=5):
  fig, ax = plt.subplots(figsize=(length,width))
  sns.violinplot(x=x,y=y,hue=hue,split=split,scale=scale,ax=ax)
  ax.set_title(f'ViolinPlot of {x.name} and {y.name}')
  ax.set_xlabel(x.name)
  ax.set_ylabel(y.name)
  plt.legend(title=hue.name)
  plt.show()

def barchart(x,y,length=10,width=5):
  fig, ax = plt.subplots(figsize=(length,width))
  sns.barplot(x=x,y=y,ax=ax)
  ax.set_title(f'Bar Chart of {x.name} and {y.name}')
  ax.set_xlabel(x.name)
  ax.set_ylabel(y.name)
  plt.show()

def stack_barchart(x,y,hue,length=10,width=5):
  fig, ax = plt.subplots(figsize=(length,width))
  sns.barplot(x=x,y=y,hue=hue,ax=ax)
  ax.set_title(f'Bar Chart of {x.name} and {y.name}')
  ax.set_xlabel(x.name)
  ax.set_ylabel(y.name)
  plt.legend(title=hue.name)
  plt.show()

from sklearn.cluster import KMeans
def kmeans_clustering(df: pd.DataFrame, n_clusters: int) -> List:
  df = df.copy()
  df.fillna(0, inplace=True)
  kmeans = KMeans(n_clusters=n_clusters)
  kmeans.fit(df)
  labels = kmeans.labels_
  centroids = kmeans.cluster_centers_
  return labels, centroids

# plt.scatter(df["PCell_RSRP_max"], df["PCell_RSRQ_max"], c=labels)
# plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", c="red")
# plt.show()

class Baseline_score():
  def __init__(
      self,
      train: pd.DataFrame,
      model: Union[XGBRegressor, LGBMRegressor, CatBoostRegressor,RandomForestRegressor],
      scaler: Union[StandardScaler, MinMaxScaler, None] = None,
      split_method: Literal["KFold", "train_test_split"] = "KFold",
      return_baseline: Literal[True, False] = True,
      num_split = 5,
      random_seed = 42,
      VERBOSE: int = 0
  ):
    if not isinstance(model, (XGBRegressor, LGBMRegressor, CatBoostRegressor,RandomForestRegressor)):
      raise TypeError("model must be a class of either XGBRegressor, LGBMRegressor, CatBoostRegressor, or LinearRegression")
    if scaler is not None and not isinstance(scaler, (StandardScaler, MinMaxScaler)):
        raise TypeError("scaler must be a class of either StandardScaler, MinMaxScaler or None")
    if not isinstance(train, pd.DataFrame):
      raise TypeError("train must be a DataFrame")

    self.train = train
    self.model = model
    self.scaler = scaler
    self.split = split_method
    self.num_split = num_split
    self.seed = random_seed
    self.return_baseline = return_baseline
    self.VERBOSE = VERBOSE
    self.feature_importances = None
    self.bsv = {}
    random.seed(random_seed)
    np.random.seed(random_seed)

  def run(self)-> float:
    X = self.train.iloc[:,:-1]
    self.feature_importances = np.zeros(X.shape[1])
    y = self.train.iloc[:,-1]

    if self.split == "train_test_split":
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)

      if self.scaler != None:
        X_train, X_test = self.scaler.fit_transform(X_train), self.scaler.fit_transform(X_test)

      if isinstance(self.model, LGBMRegressor):
        self.model.fit(X_train, y_train, eval_set=(X_test, y_test), eval_metric='rmse', callbacks=[early_stopping(stopping_rounds=100)], verbose=self.VERBOSE)

      if isinstance(self.model, CatBoostRegressor):
        self.model.fit(X_train,y_train,eval_set=(X_test,y_test),use_best_model=True,early_stopping_rounds=20,verbose=self.VERBOSE)

      else:
        self.model.fit(X_train,y_train)

      self.feature_importances += self.model.feature_importances_
      results = self.model.best_score_
      preds = self.model.predict(X_test)
      train_baseline = self.model.predict(X)
      self.bsv.update({'train':train_baseline})

      if self.VERBOSE > 0:
        print(f"split: {(self.split).upper()}")
        print(f"RMSE {mean_squared_error(y_test,preds,squared=False)}")
        print(f"Model RMSE: {results.values()}")

      score = mean_squared_error(y_test,preds,squared=False)

      if self.return_baseline:
        return score, self.bsv

      return score

    else:
      fold = KFold(n_splits=self.num_split)
      oof_f1 = []
      i = 1
      for train_index, test_index in fold.split(X, y):
        X_train, X_test = X.iloc[train_index],X.iloc[test_index]
        y_train, y_test = y.iloc[train_index],y.iloc[test_index]

        if self.scaler != None:
          X_train, X_test = self.scaler.fit_transform(X_train), self.scaler.fit_transform(X_test)

        if isinstance(self.model, LGBMRegressor):
          self.model.fit(X_train, y_train, eval_set=(X_test, y_test),eval_metric='rmse',callbacks=[early_stopping(stopping_rounds=20)],verbose=self.VERBOSE)

        if isinstance(self.model, CatBoostRegressor):
          self.model.fit(X_train,y_train,eval_set=(X_test,y_test),use_best_model=True,early_stopping_rounds=20,verbose=self.VERBOSE)

        else:
          self.model.fit(X_train,y_train)

        self.feature_importances += self.model.feature_importances_
        results = self.model.best_score_
        preds=self.model.predict(X_test)
        train_baseline = self.model.predict(X)
        self.bsv.update({f'{self.split}{i}train':train_baseline})

        if self.VERBOSE > 0:
          print(f"split {self.split}{i}")
          print(f"RMSE {mean_squared_error(y_test,preds,squared=False)}")
          print(f"Model RMSE: {results.values()}")
        oof_f1.append(mean_squared_error(y_test,preds,squared=False))
        i += 1

      self.feature_importances = self.feature_importances / self.num_split
      score = sum(oof_f1)/self.num_split

      if self.return_baseline:
        return score, self.bsv

      return score

  def plot_features(self, n_features=20):
    X = self.train.iloc[:,:-1]
    features = pd.DataFrame({'feature': list(X.columns),
                             'importance': self.feature_importances}).sort_values('importance', ascending = True)
    norm = plt.Normalize(features['importance'][:n_features].min(), features['importance'][:n_features].max())
    cmap = cm.ScalarMappable(norm=norm, cmap='Reds')
    colors = [cmap.to_rgba(val) for val in features['importance'][:n_features].values]
    plt.figure(figsize=(25,10))
    plt.barh(features['feature'][:n_features],features['importance'][:n_features], color=colors)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.show()

class Baseline_predict(Baseline_score):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.preds = None

  def predict(self, test):
    X = self.train.iloc[:,:-1]
    y = self.train.iloc[:,-1]

    if self.split == "train_test_split":
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=self.seed)
      if self.scaler != None:
        X_train, X_test = self.scaler.fit_transform(X_train), self.scaler.fit_transform(X_test)

      if isinstance(self.model, LGBMRegressor):
        self.model.fit(X_train, y_train, eval_set=(X_test, y_test), eval_metric='rmse', callbacks=[early_stopping(stopping_rounds=100)], verbose=self.VERBOSE)

      if isinstance(self.model, CatBoostRegressor):
        self.model.fit(X_train,y_train,eval_set=(X_test,y_test),use_best_model=True,early_stopping_rounds=20,verbose=self.VERBOSE)
        
      else:
        self.model.fit(X_train,y_train)

      self.preds = self.model.predict(test)
      print(f"split: {(self.split).upper()}")
      return self.preds

    else:
      fold = KFold(n_splits=self.num_split)
      oof_preds = []
      i = 1
      for train_index, test_index in fold.split(X, y):
        X_train, X_test = X.iloc[train_index],X.iloc[test_index]
        y_train, y_test = y.iloc[train_index],y.iloc[test_index]
        if self.scaler != None:
          X_train, X_test = self.scaler.fit_transform(X_train), self.scaler.fit_transform(X_test)

        if isinstance(self.model, LGBMRegressor):
          self.model.fit(X_train, y_train, eval_set=(X_test, y_test), eval_metric='rmse', callbacks=[early_stopping(stopping_rounds=100)], verbose=self.VERBOSE)

        if isinstance(self.model, CatBoostRegressor):
          self.model.fit(X_train,y_train,eval_set=(X_test,y_test),use_best_model=True,early_stopping_rounds=20,verbose=self.VERBOSE)

        else:
          self.model.fit(X_train,y_train)

        pred=self.model.predict(test)
        train_baseline = self.model.predict(test)
        self.bsv.update({f'{self.split}{i}train':train_baseline})
        if self.VERBOSE > 0:
          print(f"split {self.split}{i}")
        oof_preds.append(pred)
        i += 1

      self.preds = oof_preds
      if self.return_baseline:
        return self.preds, self.bsv
      return self.preds

  def submission(self, test):
    pres = self.preds
    if self.split == 'KFold':
      final_preds = sum(pres)/self.num_split
    else:
      final_preds = pres
    predictions_df = pd.DataFrame({'id': test.id, 'target': final_preds})
    return predictions_df



