"""汇总已经固定的四方法结果与局部示例，不重新选择测试阈值或模型。"""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1];ASSETS=Path(os.environ.get('CVPROJECT_ASSETS',ROOT))
sys.path.insert(0,str(ASSETS/'part2/.deps'));sys.path.insert(0,str(ROOT/'part1'))
import numpy as np,pandas as pd,common
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    dest=ROOT/'results/part12_comparison';dest.mkdir(exist_ok=True)
    p1=pd.read_csv(ROOT/'results/part1_course448/metrics.csv');p2=pd.read_csv(ROOT/'results/part2_course448/all_runs.csv');p1=p1.rename(columns={'image_fpr_at_threshold':'test_fpr'})
    metrics=['image_auroc','image_ap','test_fpr','pixel_auroc','pixel_ap','aupro'];records=[]
    for method in ['template','autoencoder']:
        d=p1[p1.method==method];assert len(d)==6
        records.append(dict(method=method,scoring='common_heatmap_top1pct',shot='not_applicable',seeds=1,**{m:float(d[m].mean()) for m in metrics}))
    for method in ['dino','clip']:
        for shot in ([1,2,4,8] if method=='dino' else [0]):
            d=p2[(p2.method==method)&(p2.shot==shot)]
            for scoring in ['common_heatmap_top1pct','official_native']:
                g=d.groupby('seed')[metrics].mean()
                if scoring=='official_native':
                    for m,n in [('image_auroc','native_image_auroc'),('image_ap','native_image_ap'),('test_fpr','native_test_fpr')]:g[m]=d.groupby('seed')[n].mean()
                records.append(dict(method=method,scoring=scoring,shot=shot,seeds=len(g),**{m:float(g[m].mean()) for m in metrics},**{m+'_std':float(g[m].std()) if len(g)>1 else None for m in metrics}))
    f=pd.DataFrame(records);f.to_csv(dest/'macro_metrics.csv',index=False)
    # 同一幅图展示整图与GT包围框局部；选择固定类别的第一张异常，不按成绩挑选。
    out=Path(os.environ['CVPROJECT_PART2_OUTPUT'])/'full'
    for category in ['capsules','chewinggum']:
        td=pd.read_csv(out/'dino/seed_42/shot_1'/category/'test/predictions.csv');row=td[td.label=='anomaly'].iloc[0]
        tc=pd.read_csv(out/'clip'/category/'test/predictions.csv');cr=tc[tc.image==row.image].iloc[0]
        rgb=common.load_rgb_image(row.image_path);mask=common.load_mask(row.mask_path);dm=np.load(row.map_path);cm=np.load(cr.map_path)
        yy,xx=np.where(mask>0);x0=max(0,int(xx.min())-28);x1=min(448,int(xx.max())+29);y0=max(0,int(yy.min())-28);y1=min(448,int(yy.max())+29)
        fig,axes=plt.subplots(2,4,figsize=(12,6))
        for c,(a,title,cmap) in enumerate([(rgb,'Input',None),(mask,'Official mask','gray'),(dm,'DINO seed42 1-shot','inferno'),(cm,'CLIP','inferno')]):
            limits={} if cmap is None else {'vmin':0 if c==1 else a.min(),'vmax':1 if c==1 else a.max()}
            axes[0,c].imshow(a,cmap=cmap,**limits);axes[1,c].imshow(a[y0:y1,x0:x1],cmap=cmap,**limits);axes[0,c].set_title(title)
            for ax in axes[:,c]:ax.axis('off')
        fig.suptitle(f'{category}: {Path(row.image).name}; crop {x1-x0} x {y1-y0}; patch width=14 input pixels');fig.tight_layout();fig.savefig(dest/f'{category}_detail.png',dpi=130);plt.close(fig)
    illumination=ROOT/'results/part1_course448/illumination/summary.csv'
    if illumination.exists():
        s=pd.read_csv(illumination);fig,ax=plt.subplots(figsize=(6,4))
        for method,g in s.groupby('method'):ax.plot(g.factor,g.pooled_fpr,marker='o',label=method)
        ax.set(xlabel='Brightness multiplier',ylabel='Normal validation FPR (fixed threshold)',ylim=(0,1.03));ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(dest/'illumination_fpr.png',dpi=150);plt.close(fig)
    print(f.to_string(index=False))
if __name__=='__main__':main()
