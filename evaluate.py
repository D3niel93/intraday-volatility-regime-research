"""Replay stored historical predictions: no model refitting or new backtest."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,brier_score_loss,roc_auc_score

def evaluate(frame,threshold=.70):
    if not .5<=threshold<=1:
        raise ValueError("Threshold must be in [0.5,1]")
    y=frame.actual_expansive.to_numpy(int);p=frame.probability_expansive.to_numpy(float)
    if not len(y) or not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.isin(y,[0,1]).all():
        raise ValueError("Invalid binary outcomes/probabilities")
    pred=p>=.5;selected=(p>=threshold)|(p<=1-threshold)
    return dict(bars=len(y),auc=roc_auc_score(y,p) if len(np.unique(y))==2 else None,
        accuracy_pct=accuracy_score(y,pred)*100,
        balanced_accuracy_pct=balanced_accuracy_score(y,pred)*100,
        brier=brier_score_loss(y,p),selected_signals=int(selected.sum()),
        coverage_pct=selected.mean()*100,
        selective_accuracy_pct=accuracy_score(y[selected],pred[selected])*100 if selected.any() else None)

def run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(__file__).resolve().parent
    f=pd.concat([pd.read_csv(p) for p in sorted((root/"data").glob("predictions_*.csv"))],ignore_index=True)
    if f.duplicated(["year","local_date","bucket"]).any():raise ValueError("Duplicate prediction keys")
    f=f.sort_values(["local_date","bucket"],kind="stable")
    out=root/"reports";out.mkdir(exist_ok=True)
    records=[]
    for year,g in f.groupby("year"):
        records.append(dict(year=year,**evaluate(g)))
    result=pd.DataFrame(records);result.to_csv(out/"recomputed_accuracy.csv",index=False)
    rows=[dict(year=year,threshold=float(t),**evaluate(g,float(t)))
        for year,g in f.groupby("year") for t in np.arange(.5,.96,.05)]
    curve=pd.DataFrame(rows);curve.to_csv(out/"confidence_curve.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for year,g in curve.groupby("year"):
        axes[0].plot(g.threshold,g.selective_accuracy_pct,"o-",label=str(year))
        axes[1].plot(g.threshold,g.coverage_pct,"o-",label=str(year))
    for ax,title in zip(axes,["Selective accuracy, not direction","Coverage of available bars"]):
        ax.set(title=title,xlabel="Confidence threshold",ylabel="%");ax.legend();ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(out/"confidence_coverage.png",dpi=150);plt.close(fig)
    print(result.to_string(index=False))
if __name__=="__main__":run()
