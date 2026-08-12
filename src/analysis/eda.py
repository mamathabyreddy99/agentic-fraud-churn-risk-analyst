import pandas as pd

def numerical_summary(df):
    x=df.select_dtypes(include='number'); return x.describe().T if not x.empty else pd.DataFrame()

def categorical_summary(df):
    rows=[]
    for c in df.select_dtypes(exclude='number').columns:
        vc=df[c].astype(str).value_counts(dropna=False)
        rows.append({'column':c,'unique':int(df[c].nunique(dropna=True)),'top_value':vc.index[0] if len(vc) else None,'top_count':int(vc.iloc[0]) if len(vc) else 0})
    return pd.DataFrame(rows)

def recommend_visualizations(df,target=None):
    out=[]
    if target in df.columns: out.append({'type':'Target distribution','column':target,'reason':'Inspect class balance before training.'})
    for c in df.select_dtypes(include='number').columns[:4]: out.append({'type':'Histogram','column':c,'reason':f'Inspect the distribution of {c}.'})
    for c in df.select_dtypes(exclude='number').columns[:3]:
        if df[c].nunique(dropna=True)<=20: out.append({'type':'Bar chart','column':c,'reason':f'Compare category frequencies for {c}.'})
    return out[:6]
