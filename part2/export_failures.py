"""固定操作阈值下导出失败清单；不据此调整Part2参数。"""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1];ASSETS=Path(os.environ.get('CVPROJECT_ASSETS',ROOT))
sys.path.insert(0,str(ASSETS/'part2/.deps'));sys.path.insert(0,str(ROOT/'part1'))
import common
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def export():
    dest=ROOT/'results/part2_course448';out=Path(os.environ['CVPROJECT_PART2_OUTPUT'])/'full'
    metrics=pd.read_csv(dest/'all_runs.csv');failures=[];selected=[]
    for method in ['dino','clip']:
        panels=[]
        for category in common.REQUIRED_CATEGORIES:
            directory=out/'dino/seed_42/shot_1'/category if method=='dino' else out/'clip'/category
            m=metrics[(metrics.method==method)&(metrics.category==category)&(metrics.seed==42)&(metrics.shot==(1 if method=='dino' else 0))].iloc[0]
            table=pd.read_csv(directory/'test/predictions.csv');table['predicted_ng']=table.native_score>m.native_threshold
            errors=table[table.predicted_ng!=table.label.eq('anomaly')].copy()
            # 固定按路径选择，避免只挑视觉效果明显的失败；每类先FP再FN各一例。
            picks=[]
            for label in ['normal','anomaly']:
                subset=errors[errors.label==label].sort_values('image')
                if len(subset):picks.append(subset.iloc[0].image)
            for image in errors.sort_values('image').image:
                if len(picks)>=2:break
                if image not in picks:picks.append(image)
            for row in errors.itertuples():
                heat=np.load(row.map_path);mask=common.load_mask(None if pd.isna(row.mask_path) else row.mask_path);pred=heat>m.pixel_threshold
                area=float(mask.mean());intersection=int((pred&mask).sum());recall=float(intersection/mask.sum()) if mask.any() else None
                # 下面是可核对的现象分组，不把形状统计当作模型失败的因果解释。
                group='normal-image false alarm' if row.label=='normal' else ('small annotated defect (<1% pixels)' if area<.01 else 'larger annotated defect (>=1% pixels)')
                record={'method':method,'category':category,'image':row.image,'error_type':'FP' if row.label=='normal' else 'FN','native_score':row.native_score,'native_threshold':m.native_threshold,'gt_pixel_fraction':area,'predicted_pixel_fraction':float(pred.mean()),'pixel_recall':recall,'observation_group':group,'selected':row.image in picks}
                failures.append(record)
                if row.image in picks:
                    selected.append(record);panels.append((row,record,heat,mask,pred))
        assert len(panels)==12,(method,len(panels))
        fig,axes=plt.subplots(12,4,figsize=(10,27))
        for index,(row,r,heat,mask,pred) in enumerate(panels):
            for col,(a,cmap) in enumerate([(common.load_rgb_image(row.image_path),None),(mask,'gray'),(heat,'inferno'),(pred,'gray')]):
                axes[index,col].imshow(a,cmap=cmap);axes[index,col].axis('off')
            axes[index,0].set_title(f"{r['category']} {r['error_type']}: {r['native_score']:.3f} / {r['native_threshold']:.3f}\n{Path(row.image).name}",fontsize=8)
        for col,title in enumerate(['Input (score / threshold)','Ground truth','Heatmap','Pixel prediction']):
            if col:axes[0,col].set_title(title,fontsize=9)
        fig.tight_layout();fig.savefig(dest/f'{method}_failures.png',dpi=100);plt.close(fig)
    pd.DataFrame(failures).to_csv(dest/'failure_inventory.csv',index=False)
    pd.DataFrame(selected).to_csv(dest/'failure_cases_24.csv',index=False)
    print('Failure inventory:',len(failures),'selected:',len(selected))

if __name__=='__main__':export()
