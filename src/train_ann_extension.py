from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning
from split_utils import DROP_FROM_X, TARGET, GROUP, CV_SEED, load_or_create_manifest, indices_from_manifest


def make_dense_prep(X):
    cat=X.select_dtypes(include=['object','category']).columns.tolist(); num=[c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num),
        ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat),
    ],sparse_threshold=0)


def main(data: Path, outdir: Path):
    outdir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(data); X=df.drop(columns=DROP_FROM_X); y=df[TARGET].astype(int); g=df[GROUP]
    manifest=load_or_create_manifest(df,outdir.parent/'tuning'/'locked_split_manifest.csv'); dev,_=indices_from_manifest(manifest)
    Xd,yd,gd=X.iloc[dev].copy(),y.iloc[dev].copy(),g.iloc[dev].copy()
    cfg={'hidden_layer_sizes':(32,16),'alpha':0.001,'learning_rate_init':0.001}
    pipe=Pipeline([('prep',make_dense_prep(Xd)),('model',MLPClassifier(**cfg,activation='relu',solver='adam',max_iter=50,early_stopping=False,random_state=CV_SEED))])
    cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=CV_SEED); rows=[]
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    for fold,(tr,va) in enumerate(cv.split(Xd,yd,gd),1):
        m=clone(pipe); m.fit(Xd.iloc[tr],yd.iloc[tr]); p=m.predict(Xd.iloc[va]); prob=m.predict_proba(Xd.iloc[va])[:,1]; yy=yd.iloc[va]
        rows.append({'fold':fold,'accuracy':accuracy_score(yy,p),'precision':precision_score(yy,p,zero_division=0),'recall':recall_score(yy,p,zero_division=0),'f1':f1_score(yy,p,zero_division=0),'roc_auc':roc_auc_score(yy,prob),'pr_auc':average_precision_score(yy,prob)})
    folds=pd.DataFrame(rows); folds.to_csv(outdir/'ann_cv_results.csv',index=False)
    metrics={'note':'Exploratory sklearn MLP extension evaluated only on development data with 3-fold student-group-aware CV; it is not a primary-model candidate and does not access locked-holdout labels. The optional Keras script demonstrates course-aligned BatchNorm/Dropout/EarlyStopping separately.','evaluation_scope':'development_3fold_group_cv_only','final_holdout_labels_accessed':False,'params':{**cfg,'hidden_layer_sizes':list(cfg['hidden_layer_sizes'])}}
    for c in ['accuracy','precision','recall','f1','roc_auc','pr_auc']:
        metrics[c+'_mean']=float(folds[c].mean()); metrics[c+'_std']=float(folds[c].std(ddof=1))
    (outdir/'ann_extension_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8'); print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('data',type=Path); p.add_argument('--output-dir',type=Path,default=Path('reports/ann_extension')); a=p.parse_args(); main(a.data,a.output_dir)
