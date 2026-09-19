"""Reproduce Assignment 2: python investigate.py data.csv
Requires Python 3 with pandas, numpy, and matplotlib.
Preserves all source rows and literal N/A and UNKNOWN labels.
Outputs profile.json and four PNG charts to analysis-output/.
"""
import sys, json, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

source = Path(sys.argv[1] if len(sys.argv) > 1 else 'data.csv')
out = Path('analysis-output')
out.mkdir(exist_ok=True)
raw = pd.read_csv(source, keep_default_na=False, dtype=str)
df = pd.read_csv(source, keep_default_na=False, na_values=[''])
numeric = list(df.select_dtypes(include='number').columns)
profile = {
    'source_file': source.name, 'bytes': source.stat().st_size,
    'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'rows': len(df), 'columns': len(df.columns),
    'duplicate_excess': int(raw.duplicated().sum()),
    'duplicate_members': int(raw.duplicated(keep=False).sum()),
    'make_model_year_duplicate_excess': int(raw.duplicated(['Make','Model','Year']).sum()),
    'missing_rule': 'Empty CSV fields only; literal N/A and UNKNOWN retained separately.',
    'variables': {}, 'outliers': {}, 'numeric_summary': {},
    'software': {'python': sys.version.split()[0], 'pandas': pd.__version__,
                 'numpy': np.__version__, 'matplotlib': matplotlib.__version__}
}
for c in df.columns:
    nonblank = raw.loc[raw[c] != '', c]
    profile['variables'][c] = {
        'pandas_dtype': str(df[c].dtype), 'blank_count': int((raw[c] == '').sum()),
        'distinct_nonblank': int(nonblank.nunique()),
        'values': sorted(nonblank.unique().tolist()),
        'counts': {str(k): int(v) for k,v in nonblank.value_counts().items()}
    }
for c in numeric:
    s = df[c].dropna()
    q1, q3 = s.quantile([.25,.75], interpolation='linear')
    lo, hi = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
    profile['outliers'][c] = {'q1': float(q1), 'q3': float(q3),
        'lower_fence': float(lo), 'upper_fence': float(hi),
        'below': int((s<lo).sum()), 'above': int((s>hi).sum()),
        'count': int(((s<lo)|(s>hi)).sum()), 'valid_n': len(s)}
    profile['numeric_summary'][c] = {k:float(v) for k,v in s.describe().items()}
pairs = df[['Engine HP','MSRP']].dropna()
profile['pearson_hp_msrp_raw'] = float(pairs.corr().iloc[0,1])
profile['scatter_n'] = len(pairs)
profile['price_floor_count'] = int((df.MSRP==2000).sum())
profile['brand_counts'] = {k:int(v) for k,v in df.Make.value_counts().items()}
profile['size_summary'] = {
    k:{'n':len(g),'median':float(g.MSRP.median())}
    for k,g in df.groupby('Vehicle Size')
}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
    'figure.facecolor':'#131820','axes.facecolor':'#131820',
    'text.color':'#f4f6fa','axes.labelcolor':'#d6dfeb',
    'xtick.color':'#c4cfdd','ytick.color':'#c4cfdd',
    'axes.edgecolor':'#536071','grid.color':'#394351',
    'savefig.facecolor':'#131820','axes.spines.top':False,
    'axes.spines.right':False})
def figure():
    fig, ax = plt.subplots(figsize=(10,5.3), layout='constrained')
    ax.set_axisbelow(True)
    ax.grid(axis='y',alpha=.5)
    return fig,ax
def save(fig,name):
    fig.savefig(out/name,dpi=170)
    plt.close(fig)
money = FuncFormatter(lambda x,p: f'${x/1e6:g}m' if x>=1e6 else f'${x/1000:g}k')
fig,ax=figure()
edges=np.geomspace(df.MSRP.min(),df.MSRP.max(),31)
counts,_=np.histogram(df.MSRP,bins=edges)
profile['histogram']={'edges':edges.tolist(),'counts':counts.tolist()}
ax.hist(df.MSRP,bins=edges,color='#c4f568',edgecolor='#131820',linewidth=.7)
ax.set_xscale('log');ax.xaxis.set_major_formatter(money)
ax.set_xticks([2000,10000,50000,200000,1000000,2000000])
ax.set_xlabel('Recorded MSRP (USD interpretation; logarithmic scale)')
ax.set_ylabel('Number of records')
ax.set_title('Recorded prices have a long upper tail',loc='left',pad=15,fontweight='bold')
save(fig,'01-price-histogram.png')
fig,ax=figure()
top=df.Make.value_counts().head(10).sort_values()
ax.grid(False);ax.grid(axis='x',alpha=.5)
ax.barh(top.index,top.values,color='#8ac8ff')
for i,v in enumerate(top.values): ax.text(v+12,i,f'{v:,}',va='center',fontsize=10)
ax.set_xlim(0,top.max()*1.16)
ax.set_xlabel('Number of records (not vehicle sales)')
ax.set_title('Ten most represented brands',loc='left',pad=15,fontweight='bold')
save(fig,'02-brand-counts.png')
fig,ax=figure()
sizes=['Compact','Midsize','Large']
ax.boxplot([df.loc[df['Vehicle Size']==s,'MSRP'] for s in sizes],
    tick_labels=[f'{s}\n(n={len(df[df["Vehicle Size"]==s]):,})' for s in sizes],
    patch_artist=True,whis=1.5,
    boxprops={'facecolor':'#c4f568','edgecolor':'#c4f568'},
    medianprops={'color':'#17220a','linewidth':2},
    whiskerprops={'color':'#c4cfdd'},capprops={'color':'#c4cfdd'},
    flierprops={'marker':'.','markersize':4,'markeredgecolor':'#8ac8ff','alpha':.45})
ax.set_yscale('log');ax.yaxis.set_major_formatter(money)
ax.set_ylabel('Recorded MSRP (USD interpretation; logarithmic scale)')
ax.set_xlabel('Vehicle size')
ax.set_title('Median recorded price increases with size',loc='left',pad=15,fontweight='bold')
save(fig,'03-price-boxplot.png')
fig,ax=figure()
ax.scatter(pairs['Engine HP'],pairs.MSRP,s=9,alpha=.22,color='#c4f568',edgecolors='none')
ax.set_yscale('log');ax.yaxis.set_major_formatter(money)
ax.set_xlabel('Engine horsepower (HP)')
ax.set_ylabel('Recorded MSRP (USD interpretation; logarithmic scale)')
ax.set_title('Higher horsepower generally accompanies higher prices',loc='left',pad=15,fontweight='bold')
save(fig,'04-horsepower-price.png')
(out/'profile.json').write_text(json.dumps(profile,indent=2,allow_nan=False))
print(json.dumps({'rows':len(df),'columns':len(df.columns),'outliers':profile['outliers'],
    'histogram_records':int(counts.sum()),'scatter_n':len(pairs)},indent=2))
