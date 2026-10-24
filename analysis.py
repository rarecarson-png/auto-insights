from pathlib import Path
import json, hashlib, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path.cwd(); OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
raw=pd.read_csv(ROOT/'data.csv')
d=raw.drop_duplicates().copy()
d=d.loc[d.MSRP.notna() & (d.MSRP>0)].reset_index(drop=True)
num=['Year','Engine HP','Engine Cylinders']
cat=['Make','Vehicle Size','Vehicle Style','Transmission Type','Driven_Wheels']
X=d[num+cat]; y=d.MSRP; groups=d.Make+'::'+d.Model
tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(X,y,groups))
assert not set(groups.iloc[tr]) & set(groups.iloc[te])
def prep():
    return ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),num),('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False)),cat)])
models={
 'Linear Regression':LinearRegression(), 'Ridge':Ridge(alpha=10),
 'Lasso':Lasso(alpha=.001,max_iter=5000), 'Elastic Net':ElasticNet(alpha=.001,l1_ratio=.5,max_iter=5000),
 'Bayesian Ridge':BayesianRidge(), 'Huber':HuberRegressor(max_iter=1000),
 'K Nearest Neighbors':KNeighborsRegressor(n_neighbors=7),
 'Decision Tree':DecisionTreeRegressor(max_depth=12,min_samples_leaf=5,random_state=42),
 'Random Forest':RandomForestRegressor(n_estimators=100,min_samples_leaf=2,random_state=42,n_jobs=2),
 'Extra Trees':ExtraTreesRegressor(n_estimators=100,min_samples_leaf=2,random_state=42,n_jobs=2),
 'Gradient Boosting':GradientBoostingRegressor(n_estimators=150,max_depth=3,random_state=42),
 'Histogram Gradient Boosting':HistGradientBoostingRegressor(max_iter=150,l2_regularization=1,random_state=42),
 'AdaBoost':AdaBoostRegressor(n_estimators=100,learning_rate=.05,random_state=42),
 'RBF Support Vector':SVR(C=10,epsilon=.05),
 'Linear Support Vector':LinearSVR(C=1,epsilon=.05,max_iter=10000,random_state=42),
 'Bagging Trees':BaggingRegressor(n_estimators=50,random_state=42,n_jobs=2)}
cv=list(GroupKFold(n_splits=3).split(X.iloc[tr],y.iloc[tr],groups.iloc[tr]))
rows=[]; fitted={}; predictions=pd.DataFrame({'actual':y.iloc[te]})
for name,model in models.items():
    start=time.time()
    pipe=TransformedTargetRegressor(regressor=make_pipeline(prep(),model),func=np.log1p,inverse_func=np.expm1)
    scores=-cross_val_score(pipe,X.iloc[tr],y.iloc[tr],cv=cv,scoring='neg_mean_absolute_error',n_jobs=1)
    pipe.fit(X.iloc[tr],y.iloc[tr]); pred=pipe.predict(X.iloc[te])
    rows.append(dict(model=name,cv_mae=scores.mean(),cv_sd=scores.std(),test_mae=mean_absolute_error(y.iloc[te],pred),test_rmse=np.sqrt(mean_squared_error(y.iloc[te],pred)),test_r2=r2_score(y.iloc[te],pred),train_mae=mean_absolute_error(y.iloc[tr],pipe.predict(X.iloc[tr])),seconds=time.time()-start))
    fitted[name]=pipe; predictions[name]=pred
    print(name, json.dumps(rows[-1]),flush=True)
metrics=pd.DataFrame(rows).sort_values('cv_mae'); metrics.to_csv(OUT/'model_metrics.csv',index=False)
predictions.to_csv(OUT/'test_predictions.csv',index=True)
winner=metrics.iloc[0]; best=winner['model']
baseline=DummyRegressor(strategy='median').fit(X.iloc[tr],y.iloc[tr]); bp=baseline.predict(X.iloc[te])
bmae=mean_absolute_error(y.iloc[te],bp)
split=d[['Make','Model','Year']].copy();split['split']='train';split.loc[te,'split']='test';split.to_csv(OUT/'split.csv',index=False)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(name):
    plt.tight_layout();plt.savefig(OUT/name,dpi=140,bbox_inches='tight');plt.close()
valid=d[['Engine HP','MSRP']].dropna(); rho,p=spearmanr(valid['Engine HP'],valid.MSRP)
plt.figure(figsize=(8,4));plt.scatter(valid['Engine HP'],valid.MSRP,s=7,alpha=.3);plt.yscale('log');plt.xlabel('Engine horsepower');plt.ylabel('MSRP (USD, log scale)');save('horsepower.png')
sizes=d.groupby('Vehicle Size').MSRP.agg(['count','median','mean']);sizes.to_csv(OUT/'size_summary.csv')
h,kp=kruskal(*(g.MSRP for _,g in d.groupby('Vehicle Size')))
plt.figure(figsize=(8,4));plt.boxplot([g.MSRP for _,g in d.groupby('Vehicle Size')],tick_labels=list(sizes.index),showfliers=False);plt.ylabel('MSRP (USD)');save('size.png')
mpg=d[['Engine HP','highway MPG']].dropna(); mr,mp=spearmanr(mpg['Engine HP'],mpg['highway MPG'])
plt.figure(figsize=(8,4));plt.scatter(mpg['Engine HP'],mpg['highway MPG'],s=7,alpha=.3);plt.xlabel('Engine horsepower');plt.ylabel('Highway MPG');save('mpg.png')
years=d.groupby('Year').MSRP.agg(['count','median']);years.to_csv(OUT/'year_summary.csv')
fig,axes=plt.subplots(2,1,figsize=(8,5),sharex=True);axes[0].plot(years.index,years['median'],marker='o');axes[0].set_ylabel('Median MSRP (USD)');axes[1].bar(years.index,years['count']);axes[1].set_ylabel('Records');axes[1].set_xlabel('Model year');save('years.png')
fig,ax=plt.subplots(figsize=(9,7));ax.barh(metrics.model,metrics.cv_mae,xerr=metrics.cv_sd,color='#286b9f');ax.invert_yaxis();ax.set_xlabel('Training group CV MAE (USD); error bars = fold SD');save('models.png')
fig,ax=plt.subplots(figsize=(8,4));ax.scatter(y.iloc[te],predictions[best],s=8,alpha=.35);lo=1;hi=max(y.iloc[te].max(),predictions[best].max());ax.plot([lo,hi],[lo,hi],'k--');ax.set(xscale='log',yscale='log',xlabel='Actual MSRP (USD, log scale)',ylabel='Predicted MSRP (USD, log scale)',title=best);save('predictions.png')
summary=dict(raw_n=len(raw),analysis_n=len(d),duplicates_removed=len(raw)-len(raw.drop_duplicates()),train_n=len(tr),test_n=len(te),train_groups=groups.iloc[tr].nunique(),test_groups=groups.iloc[te].nunique(),hp_rho=rho,hp_p=p,hp_n=len(valid),size_h=h,size_p=kp,mpg_rho=mr,mpg_p=mp,mpg_n=len(mpg),best=best,baseline_mae=bmae,sklearn=sklearn.__version__,sha256=hashlib.sha256((ROOT/'data.csv').read_bytes()).hexdigest())
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print('SUMMARY',json.dumps(summary),flush=True)
