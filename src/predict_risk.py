from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from recommendation_engine import generate_recommendations


def raw_score(model, X):
    if hasattr(model,'decision_function'):
        return np.asarray(model.decision_function(X),dtype=float)
    p=np.clip(model.predict_proba(X)[:,1],1e-6,1-1e-6)
    return np.log(p/(1-p))

def level(p,low,high):
    return 'High' if p>=high else ('Medium' if p>=low else 'Low')

def main(bundle_path,input_csv,output_csv):
    b=joblib.load(bundle_path);df=pd.read_csv(input_csv)
    missing=[c for c in b['feature_columns'] if c not in df.columns]
    if missing: raise ValueError(f'Missing required feature columns: {missing}')
    X=df[b['feature_columns']].copy();score=raw_score(b['base_model'],X);prob=b['calibrator'].predict_proba(score.reshape(-1,1))[:,1]
    levels=[level(float(p),b['low_threshold'],b['high_threshold']) for p in prob]
    recs=[]
    for (_,r),lv in zip(df.iterrows(),levels):
        rr=generate_recommendations(r,b['recommendation_config']) if lv!='Low' else [{'factor':'monitor','recommendation':'Continue routine monitoring and reinforce current study habits.','reason':'Risk probability is below the low-risk boundary.'}]
        recs.append(json.dumps(rr,ensure_ascii=False))
    out=df.copy();out['risk_probability']=prob;out['risk_level']=levels;out['predicted_at_risk']=(prob>=b['high_threshold']).astype(int);out['recommendations_json']=recs;out.to_csv(output_csv,index=False);print(f'Wrote {len(out)} predictions to {output_csv}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);p.add_argument('input_csv',type=Path);p.add_argument('output_csv',type=Path);a=p.parse_args();main(a.bundle,a.input_csv,a.output_csv)
