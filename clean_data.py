"""Auto Insights Assignment 3. Run: python clean_data.py data.csv --out prepared
Requires pandas and numpy. Raw input is never modified. CSV blank = missing.
source_row_id is the 1-based data-record number (header excluded).
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

NAMES = ['make','model','year','fuel_type','engine_hp','engine_cylinders',
         'transmission','driven_wheels','doors','market_category','vehicle_size',
         'vehicle_style','highway_mpg','city_mpg','popularity','msrp']
NUMERIC = ['year','engine_hp','engine_cylinders','doors','highway_mpg','city_mpg','popularity','msrp']

def run(source, out):
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    expected = ['Make','Model','Year','Engine Fuel Type','Engine HP','Engine Cylinders','Transmission Type','Driven_Wheels','Number of Doors','Market Category','Vehicle Size','Vehicle Style','highway MPG','city mpg','Popularity','MSRP']
    assert raw.columns.tolist() == expected, 'Unexpected source schema'
    x = raw.copy(); x.columns = NAMES
    x.insert(0,'source_row_id',np.arange(1,len(x)+1))
    log = []
    def change(idx,col,new,reason):
        old = x.at[idx,col]
        log.append({'source_row_id':int(x.at[idx,'source_row_id']), 'column':col,
                    'original_value':str(old),'prepared_value':'' if pd.isna(new) else str(new),'reason':reason})
        x.at[idx,col] = new
    dup = raw.duplicated(keep='first')
    first = {}
    removed=[]
    for idx,row in raw.iterrows():
        key=tuple(row)
        if key in first: removed.append({'removed_source_row_id':idx+1,'kept_source_row_id':first[key]})
        else: first[key]=idx+1
    pd.DataFrame(removed,columns=['removed_source_row_id','kept_source_row_id']).to_csv(out/'duplicates_removed.csv',index=False)
    x = x.loc[~dup].copy()
    raw_blanks={n:int((raw[c]=='').sum()) for c,n in zip(expected,NAMES)}
    kept_blanks={c:int(x[c].eq('').sum()) for c in NAMES}
    # Whitespace normalization and explicit null semantics.
    for c in NAMES:
        for idx,v in x[c].items():
            if v != v.strip(): change(idx,c,v.strip(),'Trim surrounding whitespace')
        for idx in x.index[x[c].eq('')]: change(idx,c,pd.NA,'Source blank retained as missing')
    for col,token in [('market_category','N/A'),('transmission','UNKNOWN')]:
        for idx in x.index[x[col].eq(token).fillna(False)]:
            change(idx,col,pd.NA,'Unspecified source label treated as missing; not a measured category')
    # Make/model identity and AWD vs 4WD distinctions are preserved.
    for col in ['fuel_type','transmission','driven_wheels','vehicle_size','vehicle_style']:
        for idx,v in x[col].dropna().items():
            new=' '.join(v.lower().replace('_',' ').split())
            if new!=v: change(idx,col,new,'Standardize category case, underscores, and spacing')
    for idx,v in x['market_category'].dropna().items():
        new='|'.join(sorted(set(t.strip().lower() for t in v.split(','))))
        if new!=v: change(idx,'market_category',new,'Canonical sorted multi-label tokens; pipe-delimited')
    failures={}
    for col in NUMERIC:
        converted=pd.to_numeric(x[col],errors='coerce')
        bad=x[col].notna() & converted.isna()
        failures[col]=int(bad.sum())
        for idx in x.index[bad]: change(idx,col,pd.NA,'Non-numeric token rejected')
        fractional=converted.notna() & (converted%1!=0)
        for idx in x.index[fractional]: change(idx,col,pd.NA,'Non-integer token rejected for whole-number field')
        x[col]=pd.to_numeric(x[col],errors='coerce').astype('Int64')
    invalid_counts={}
    rules={
        'year':x.year.notna() & ~x.year.between(1886,2027),
        'engine_hp':x.engine_hp.notna() & (x.engine_hp<=0),
        'engine_cylinders':x.engine_cylinders.notna() & (x.engine_cylinders<0),
        'doors':x.doors.notna() & (x.doors<=0),
        'highway_mpg':x.highway_mpg.notna() & (x.highway_mpg<=0),
        'city_mpg':x.city_mpg.notna() & (x.city_mpg<=0),
        'popularity':x.popularity.notna() & (x.popularity<0),
        'msrp':x.msrp.notna() & (x.msrp<=0)}
    for c,bad in rules.items():
        invalid_counts[c]=int(bad.sum())
        for idx in x.index[bad.fillna(False)]: change(idx,c,pd.NA,'Failed documented range validation')
    # Conservative review rule, not a claimed correction from an external source.
    suspect=(x.fuel_type.notna() & x.fuel_type.ne('electric') & x.highway_mpg.gt(150)).fillna(False)
    x['highway_mpg_review_flag']=suspect
    for idx in x.index[suspect]:
        change(idx,'highway_mpg',pd.NA,'Non-electric highway value >150 quarantined pending source verification; no replacement guessed')
    x['electric_efficiency_review_flag']=x.fuel_type.eq('electric').fillna(False)
    x['msrp_floor_review_flag']=x.msrp.eq(2000).fillna(False)
    missing_after={c:int(x[c].isna().sum()) for c in NAMES}
    for c in ['fuel_type','engine_hp','engine_cylinders','doors','market_category','transmission','highway_mpg']:
        x[c+'_missing']=x[c].isna()
    bounds={}
    for c in ['msrp','engine_hp','engine_cylinders','city_mpg','highway_mpg']:
        q1,q3=x[c].quantile([.25,.75]).astype(float);iqr=q3-q1
        low,high=q1-1.5*iqr,q3+1.5*iqr
        flag=((x[c]<low)|(x[c]>high)).fillna(False)
        x[c+'_iqr_outlier']=flag
        bounds[c]={'q1':q1,'q3':q3,'lower_fence':low,'upper_fence':high,'flagged':int(flag.sum())}
    x['log10_msrp']=np.log10(x.msrp.astype(float))
    x['core_analysis_ready']=x[['make','model','year','engine_hp','msrp']].notna().all(axis=1)
    x['fuel_economy_analysis_ready']=x[['fuel_type','city_mpg','highway_mpg']].notna().all(axis=1)&~x.electric_efficiency_review_flag
    # Required gates: provenance, reconciliation, integer types and transformations.
    assert len(raw)==len(x)+len(removed)
    assert x.source_row_id.is_unique
    assert not x[NAMES].duplicated().any(), 'New duplicates after normalization require review'
    assert all(str(x[c].dtype)=='Int64' for c in NUMERIC)
    assert np.isfinite(x.log10_msrp).all()
    assert np.allclose(10**x.log10_msrp,x.msrp.astype(float))
    x.to_csv(out/'data_cleaned.csv',index=False,float_format='%.10f')
    pd.DataFrame(log).to_csv(out/'cleaning_changes.csv',index=False)
    descriptions={c:'Prepared source field; see cleaning decisions and original data dictionary.' for c in NAMES}
    descriptions.update({'source_row_id':'Unique original CSV data-record number; header excluded. Not a vehicle ID.', 'log10_msrp':'Base-10 logarithm of original positive MSRP; dollar values retained.', 'core_analysis_ready':'Nonmissing make, model, year, engine_hp, msrp; pairwise or other models need their own masks.', 'fuel_economy_analysis_ready':'Known non-electric fuel and nonmissing city/highway efficiency; conservative unit-comparability subset.', 'highway_mpg_review_flag':'Non-electric highway >150 was set missing for analysis; see change log.', 'electric_efficiency_review_flag':'Electric record: efficiency unit needs source verification before comparison with gasoline MPG.', 'msrp_floor_review_flag':'MSRP equals source minimum 2000; price construction needs verification.'})
    dictionary=[]
    for c in x:
        desc=descriptions.get(c,'Missingness flag: true means missing.' if c.endswith('_missing') else 'Outside 1.5 IQR fences on prepared nonmissing values. Review only; retained.')
        dictionary.append({'column':c,'python_dtype':str(x[c].dtype),'missing':int(x[c].isna().sum()),'description':desc})
    pd.DataFrame(dictionary).to_csv(out/'cleaned_dictionary.csv',index=False)
    report={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_bytes':source.stat().st_size,'input_rows':len(raw),'original_columns':len(NAMES),'exact_duplicates_removed':len(removed),'prepared_rows':len(x),'prepared_columns':len(x.columns),'raw_blanks':raw_blanks,'retained_blanks_before_cleaning':kept_blanks,'missing_after':missing_after,'parse_failures':failures,'invalid_range_counts':invalid_counts,'suspect_highway_values_quarantined':int(suspect.sum()),'electric_records_flagged':int(x.electric_efficiency_review_flag.sum()),'msrp_floor_flagged':int(x.msrp_floor_review_flag.sum()),'core_ready_rows':int(x.core_analysis_ready.sum()),'fuel_economy_ready_rows':int(x.fuel_economy_analysis_ready.sum()),'outliers':bounds,'change_log_cells':len(log),'checks_passed':True,'python_pandas':pd.__version__,'numpy':np.__version__,'dictionary':dictionary}
    (out/'cleaning_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['dictionary','outliers']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--out',type=Path,default=Path('prepared'))
    a=p.parse_args();run(a.source,a.out)
