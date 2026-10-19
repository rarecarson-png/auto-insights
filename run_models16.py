import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import json,time,warnings,hashlib
import numpy as np,pandas as pd,sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit,GroupKFold
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,BayesianRidge,HuberRegressor,SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.inspection import permutation_importance
R=Path(__file__).resolve().parent;O=R/'model_results16';O.mkdir(exist_ok=True)
d=pd.read_csv(R/'data_cleaned.csv');num=['year','engine_hp','engine_cylinders'];cat=['make','vehicle_size','vehicle_style','transmission','driven_wheels'];features=num+cat
X=d[features];y=d.msrp;groups=d.make+'::'+d.model
tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(X,y,groups));assert not set(groups.iloc[tr])&set(groups.iloc[te])
models={
'Linear Regression':LinearRegression(), 'Ridge':Ridge(alpha=10), 'Lasso':Lasso(alpha=.001,max_iter=10000), 'Elastic Net':ElasticNet(alpha=.001,l1_ratio=.5,max_iter=10000), 'Bayesian Ridge':BayesianRidge(), 'Huber Regression':HuberRegressor(max_iter=5000), 'SGD Regression':SGDRegressor(alpha=.001,max_iter=3000,tol=1e-5,random_state=42), 'K-Nearest Neighbors':KNeighborsRegressor(n_neighbors=10,weights='distance'), 'Decision Tree':DecisionTreeRegressor(max_depth=8,min_samples_leaf=5,random_state=42), 'Extra Tree':ExtraTreeRegressor(max_depth=18,min_samples_leaf=2,random_state=42), 'Random Forest':RandomForestRegressor(n_estimators=120,max_depth=18,min_samples_leaf=2,random_state=42,n_jobs=2), 'Extra Trees':ExtraTreesRegressor(n_estimators=120,max_depth=18,min_samples_leaf=2,random_state=42,n_jobs=2), 'Gradient Boosting':GradientBoostingRegressor(n_estimators=150,max_depth=3,min_samples_leaf=5,random_state=42), 'Histogram Gradient Boosting':HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=20,l2_regularization=1,early_stopping=False,random_state=42), 'AdaBoost':AdaBoostRegressor(n_estimators=80,learning_rate=.05,random_state=42), 'Support Vector Regression':SVR(C=10,epsilon=.05), 'Median baseline':DummyRegressor(strategy='median')}
def pipe(m):
 prep=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),num),('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',drop='first',sparse_output=False)),cat)])
 return make_pipeline(prep,TransformedTargetRegressor(regressor=clone(m),transformer=StandardScaler()))
folds=list(GroupKFold(n_splits=3).split(X.iloc[tr],y.iloc[tr],groups.iloc[tr]));rows=[];fitted={};pred=d.iloc[te][['source_row_id','make','model','vehicle_size','msrp']].copy();foldrows=[]
for name,m in models.items():
 start=time.time();scores=[];msgs=[]
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always')
  for i,(a,b) in enumerate(folds):
   p=pipe(m);p.fit(X.iloc[tr[a]],y.iloc[tr[a]]);v=p.predict(X.iloc[tr[b]]);score=mean_absolute_error(y.iloc[tr[b]],v);scores.append(score);foldrows.append({'model':name,'fold':i+1,'mae':score,'n':len(b)})
  p=pipe(m);p.fit(X.iloc[tr],y.iloc[tr]);v=p.predict(X.iloc[te]);train=p.predict(X.iloc[tr]);fitted[name]=p;pred[name]=v
  msgs=sorted(set(str(w.message) for w in ws if 'unknown categories' not in str(w.message)))
 rows.append(dict(model=name,cv_mae=np.mean(scores),cv_std=np.std(scores,ddof=1),test_mae=mean_absolute_error(y.iloc[te],v),test_rmse=np.sqrt(mean_squared_error(y.iloc[te],v)),test_r2=r2_score(y.iloc[te],v),train_mae=mean_absolute_error(y.iloc[tr],train),negative_predictions=int((v<0).sum()),seconds=time.time()-start,warnings='; '.join(msgs)))
 pd.DataFrame(rows).to_csv(O/'metrics.csv',index=False);print(name,round(rows[-1]['cv_mae']),round(rows[-1]['test_mae']),round(time.time()-start,1),flush=True)
pd.DataFrame(foldrows).to_csv(O/'cv_folds.csv',index=False);pred.to_csv(O/'test_predictions.csv',index=False)
best=min([r for r in rows if r['model']!='Median baseline'],key=lambda r:r['cv_mae'])['model']
imp=permutation_importance(fitted[best],X.iloc[te],y.iloc[te],scoring='neg_mean_absolute_error',n_repeats=3,random_state=42);pd.DataFrame({'feature':features,'mae_increase':imp.importances_mean,'std':imp.importances_std}).to_csv(O/'importance.csv',index=False)
g=[]
for name in models:
 for size,s in pred.groupby('vehicle_size'):g.append({'model':name,'size':size,'n':len(s),'mae':mean_absolute_error(s.msrp,s[name])})
pd.DataFrame(g).to_csv(O/'group_errors.csv',index=False)
split=d[['source_row_id','make','model']].copy();split['split']='train';split.loc[te,'split']='test';split.to_csv(O/'split.csv',index=False)
summary=dict(selected_model=best,selection='Lowest mean 3-fold grouped cross-validation MAE within training data; no hyperparameter search',n=len(d),train_n=len(tr),test_n=len(te),train_groups=groups.iloc[tr].nunique(),test_groups=groups.iloc[te].nunique(),group_overlap=0,features=features,seed=42,sklearn=sklearn.__version__,sha256=hashlib.sha256((R/'data_cleaned.csv').read_bytes()).hexdigest(),parameters={n:str(m) for n,m in models.items()})
(O/'summary.json').write_text(json.dumps(summary,indent=2));print('SELECTED',best,flush=True)
