def detect_identifier_columns(df,target=None):
    out=[]; n=max(len(df),1)
    for c in df.columns:
        if c==target: continue
        name=c.lower().strip(); ratio=df[c].nunique(dropna=True)/n
        if (name=='id' or name.endswith('_id') or name.startswith('id_') or 'uuid' in name or 'guid' in name) and ratio>.5: out.append(c)
    return out

def drop_identifier_columns(df,target=None):
    ids=detect_identifier_columns(df,target); return df.drop(columns=ids,errors='ignore'),ids
