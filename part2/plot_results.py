'Plot verified few-shot curves and representative four-panel predictions.'
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1];ASSETS=Path(os.environ.get('CVPROJECT_ASSETS',ROOT));sys.path.insert(0,str(ASSETS/'part2/.deps'));sys.path.insert(0,str(ROOT/'part1'))
import common
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__=='__main__':
    dest=ROOT/'results/part2_course448';frame=pd.read_csv(dest/'all_runs.csv');dino=frame[frame.method=='dino'];clip=frame[frame.method=='clip']
    macro=dino.groupby(['shot','seed']).mean(numeric_only=True).reset_index()
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for ax,metric in zip(axes.ravel(),['native_image_auroc','native_image_ap','pixel_ap','aupro']):
        stats=macro.groupby('shot')[metric].agg(['mean','std']);ax.errorbar(stats.index,stats['mean'],yerr=stats['std'],marker='o',capsize=4,label='DINO mean +/- sample SD')
        ax.axhline(clip[metric].mean(),linestyle='--',label='CLIP zero-shot macro mean');ax.set_xticks([1,2,4,8]);ax.set_xlabel('Normal reference images');ax.set_title(metric);ax.set_ylim(0,1.03);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/'few_shot_curves.png',dpi=150);plt.close(fig)
    out=Path(os.environ['CVPROJECT_PART2_OUTPUT'])/'full'
    for model in ['dino','clip']:
        fig,axes=plt.subplots(6,4,figsize=(12,18))
        for index,category in enumerate(common.REQUIRED_CATEGORIES):
            directory=out/'dino/seed_42/shot_1'/category if model=='dino' else out/'clip'/category
            table=pd.read_csv(directory/'test/predictions.csv');row=table[table.label=='anomaly'].iloc[0]
            image=common.load_rgb_image(row.image_path);mask=common.load_mask(row.mask_path);heat=np.load(row.map_path)
            metric=frame[(frame.method==model)&(frame.category==category)&(frame.seed==42)&(frame.shot==(1 if model=='dino' else 0))]
            m=metric.iloc[0]
            for column,(panel,cmap) in enumerate([(image,None),(mask,'gray'),(heat,'inferno'),(heat>m.pixel_threshold,'gray')]):axes[index,column].imshow(panel,cmap=cmap);axes[index,column].axis('off')
            axes[index,0].set_title(f'{category}: native={row.native_score:.3f}\nthreshold={m.native_threshold:.3f}',fontsize=9)
        for column,title in enumerate(['Input','Ground truth','Heatmap','Thresholded']):
            if column:axes[0,column].set_title(title)
        fig.tight_layout();fig.savefig(dest/f'{model}_examples.png',dpi=120);plt.close(fig)
