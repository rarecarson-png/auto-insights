from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent; out=root/'model_results';out.mkdir(exist_ok=True)
d=pd.read_csv(root/'data_cleaned.csv')
num=['year','engine_hp','engine_cylinders'];cat=['make','vehicle_size','vehicle_style','transmission','driven_wheels'];features=num+cat
X=d[features];y=d.msrp;groups=d['make']+'::'+d['model']
tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(X,y,groups))
assert not set(groups.iloc[tr])&set(groups.iloc[te])
def prep():
 return ColumnTransformer([('numeric',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),num),('category',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',drop='first')),cat)])
models={'Median baseline':DummyRegressor(strategy='median'),'Linear Regression':LinearRegression(),'Decision Tree':DecisionTreeRegressor(max_depth=8,min_samples_leaf=5,random_state=42),'Random Forest':RandomForestRegressor(n_estimators=120,max_depth=18,min_samples_leaf=2,max_features=1.0,random_state=42,n_jobs=2)}
rows=[];pred=d.iloc[te][['source_row_id','make','model','year','vehicle_size','msrp']].copy();fitted={}
for name,model in models.items():
 pipe=make_pipeline(prep(),model);pipe.fit(X.iloc[tr],y.iloc[tr]);hat=pipe.predict(X.iloc[te]);trainhat=pipe.predict(X.iloc[tr]);pred[name]=hat
 rows.append({'model':name,'test_mae':mean_absolute_error(y.iloc[te],hat),'test_rmse':np.sqrt(mean_squared_error(y.iloc[te],hat)),'test_r2':r2_score(y.iloc[te],hat),'train_mae':mean_absolute_error(y.iloc[tr],trainhat),'negative_predictions':int((hat<0).sum())});fitted[name]=pipe
 print(name,round(rows[-1]['test_mae']),flush=True)
metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics.csv',index=False);pred.to_csv(out/'test_predictions.csv',index=False)
split=d[['source_row_id','make','model']].copy();split['split']='train';split.loc[te,'split']='test';split.to_csv(out/'split.csv',index=False)
imp=permutation_importance(fitted['Random Forest'],X.iloc[te],y.iloc[te],scoring='neg_mean_absolute_error',n_repeats=3,random_state=42,n_jobs=1)
importance=pd.DataFrame({'feature':features,'mae_increase':imp.importances_mean,'shuffle_std':imp.importances_std}).sort_values('mae_increase');importance.to_csv(out/'permutation_importance.csv',index=False)
group=[]
for name in models:
 for size,sub in pred.groupby('vehicle_size'):group.append({'model':name,'vehicle_size':size,'test_n':len(sub),'mae':mean_absolute_error(sub.msrp,sub[name])})
pd.DataFrame(group).to_csv(out/'group_errors.csv',index=False)
summary={'records':len(d),'train_records':len(tr),'test_records':len(te),'train_groups':groups.iloc[tr].nunique(),'test_groups':groups.iloc[te].nunique(),'group_overlap':0,'seed':42,'features':features,'sklearn':sklearn.__version__,'source_sha256':hashlib.sha256((root/'data_cleaned.csv').read_bytes()).hexdigest(),'metrics':rows}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'figure.facecolor':'#141b23','axes.facecolor':'#141b23','text.color':'#f5f7fa','axes.labelcolor':'#dbe4ed','xtick.color':'#dbe4ed','ytick.color':'#dbe4ed','axes.edgecolor':'#526270','font.size':11,'savefig.facecolor':'#141b23'})
def save(n):plt.tight_layout(pad=2);plt.savefig(out/n,dpi=140);plt.close()
fig,ax=plt.subplots(figsize=(10,4.5));b=ax.barh(metrics.model,metrics.test_mae,color=['#627186','#74c8ff','#bda2ff','#c1f769']);ax.bar_label(b,labels=[f'${v:,.0f}' for v in metrics.test_mae],padding=5,color='white');ax.set(xlabel='Held-out mean absolute error (USD); lower is better',xlim=(0,metrics.test_mae.max()*1.23));save('mae.png')
fig,axes=plt.subplots(1,3,figsize=(14,4.8))
for ax,name in zip(axes,list(models)[1:]):
 ax.scatter(pred.msrp,pred[name],s=6,alpha=.3,color='#c1f769');lo=min(pred.msrp.min(),pred[name].min());hi=max(pred.msrp.max(),pred[name].max());ax.plot([lo,hi],[lo,hi],ls='--',color='#74c8ff');ax.set_xscale('symlog',linthresh=2000);ax.set_yscale('symlog',linthresh=2000);ax.set(title=name,xlabel='Actual MSRP (USD)',ylabel='Predicted MSRP (USD)')
save('predicted_actual.png')
fig,ax=plt.subplots(figsize=(10,5));ax.scatter(pred['Random Forest'],pred['Random Forest']-pred.msrp,s=8,alpha=.3,color='#bda2ff');ax.axhline(0,color='#74c8ff',ls='--');ax.set_xscale('symlog',linthresh=2000);ax.set_yscale('symlog',linthresh=2000);ax.set(xlabel='Random Forest predicted MSRP (USD)',ylabel='Prediction − actual MSRP (USD)');save('residuals.png')
fig,ax=plt.subplots(figsize=(10,5));ax.barh(importance.feature,importance.mae_increase,xerr=importance.shuffle_std,color='#c1f769',error_kw={'ecolor':'#74c8ff'});ax.axvline(0,color='white',lw=.5);ax.set(xlabel='Increase in held-out MAE after shuffling (USD)');save('importance.png')
g=pd.DataFrame(group);fig,ax=plt.subplots(figsize=(10,5));sizes=sorted(g.vehicle_size.unique());pos=np.arange(len(sizes));width=.2
for i,name in enumerate(models):
 v=g[g.model==name].set_index('vehicle_size').loc[sizes];ax.bar(pos+(i-1.5)*width,v.mae,width,label=name)
ax.set_xticks(pos,[f'{s}\nn={int(g[g.vehicle_size==s].iloc[0].test_n)}' for s in sizes]);ax.set(ylabel='Held-out MAE (USD)');ax.legend(fontsize=9);save('group_errors.png')
print(json.dumps(summary,indent=2),flush=True)
