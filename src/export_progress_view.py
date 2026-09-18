from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

KEY=['id_student','code_module','code_presentation']
PROGRESS=[
 'clicks_week1_28d','clicks_week2_28d','clicks_week3_28d','clicks_week4_28d',
 'weekly_click_slope_28d','active_days_28d','active_days_last7d',
 'days_since_last_activity_28d','assessment_mean_score_28d',
 'assessment_completion_rate_28d','assessments_missed_28d'
]

def main(data: Path, predictions: Path, output: Path):
    df=pd.read_csv(data)
    pred=pd.read_csv(predictions)
    view=pred.merge(df[KEY+PROGRESS],on=KEY,how='left',validate='one_to_one')
    view['engagement_direction']=pd.cut(
        view['weekly_click_slope_28d'],[-float('inf'),-1e-9,1e-9,float('inf')],
        labels=['Declining','Stable','Improving'])
    output.parent.mkdir(parents=True,exist_ok=True)
    view.to_csv(output,index=False)
    print(f'Wrote {len(view)} progress rows to {output}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('predictions',type=Path);p.add_argument('output',type=Path);a=p.parse_args();main(a.data,a.predictions,a.output)
