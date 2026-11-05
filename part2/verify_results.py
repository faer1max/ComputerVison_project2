"""核对Part2完整输出并生成可分享摘要；不训练或选择测试阈值。"""
from pathlib import Path
import os,sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];ASSETS=Path(os.environ.get('CVPROJECT_ASSETS',ROOT))
sys.path.insert(0,str(ASSETS/'part2/.deps'));sys.path.insert(0,str(ROOT/'part1'))
import numpy as np,pandas as pd
import common
from sklearn.metrics import roc_auc_score,average_precision_score
OUT=Path(os.environ['CVPROJECT_PART2_OUTPUT'])/'full'
DEST=ROOT/'results/part2_course448'

def verify():
    expected={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'part2/benchmark.py',ROOT/'part2/course_inputs.py',ROOT/'experiment_protocol.json',ROOT/'part2/reference_selections.csv',ROOT/'part1/03_evaluate.py',ROOT/'part1/common.py']}
    assert json.loads((OUT/'source.json').read_text())==expected
    paths=list((OUT/'dino').glob('seed_*/shot_*/*/metrics.json'))+list((OUT/'clip').glob('*/metrics.json'))
    assert len(paths)==78,len(paths)
    required={('dino',c,s,k) for c in common.REQUIRED_CATEGORIES for s in [42,43,44] for k in [1,2,4,8]}|{('clip',c,42,0) for c in common.REQUIRED_CATEGORIES}
    seen=set()
    selections=pd.read_csv(ROOT/'part2/reference_selections.csv');records=[];resources=[];count=0
    for path in paths:
        directory=path.parent;result=json.loads(path.read_text());category=result['category'];meta=json.loads((directory/'run.json').read_text())
        key=('dino' if 'seed' in meta else 'clip',category,meta.get('seed',42),meta.get('shot',0))
        assert key in required and key not in seen;seen.add(key)
        fit,val=common.split_normal_train(common.load_visa_split(category,'train'));test=common.load_visa_split(category,'test')
        if 'seed' in meta:
            chosen=selections[(selections.category==category)&(selections.seed==meta['seed'])&(selections.shot==meta['shot'])]
            assert chosen.image.tolist()==meta['references'] and set(chosen.image)<=set(fit.image)
        tables={}
        for split,expected_rows in [('validation',val),('test',test)]:
            table=pd.read_csv(directory/split/'predictions.csv');assert set(table.image)==set(expected_rows.image) and len(table)==len(expected_rows)
            assert table.set_index('image').label.to_dict()==expected_rows.set_index('image').label.to_dict()
            for column in ['image_path','mask_path']:
                assert table.set_index('image')[column].fillna('').astype(str).to_dict()==expected_rows.set_index('image')[column].fillna('').astype(str).to_dict()
            assert set(map(Path,table.map_path))==set((directory/split/'maps').glob('*.npy'))
            assert np.isfinite(table.native_score).all()
            for row in table.itertuples():
                a=np.load(row.map_path);assert a.shape==(448,448) and a.dtype==np.float32 and np.isfinite(a).all();assert np.isclose(common.image_score(a),row.score,atol=1e-6);count+=1
            tables[split]=table
        v,t=tables['validation'],tables['test'];assert (v.label=='normal').all()
        threshold=float(np.quantile(v.native_score,.95,method='higher'));labels=t.label.eq('anomaly')
        result.update({'method':'dino' if 'seed' in meta else 'clip','seed':meta.get('seed',42),'shot':meta.get('shot',0),
           'native_image_auroc':float(roc_auc_score(labels,t.native_score)),'native_image_ap':float(average_precision_score(labels,t.native_score)),
           'native_threshold':threshold,'native_validation_fpr':float((v.native_score>threshold).mean()),'native_test_fpr':float((t.loc[~labels,'native_score']>threshold).mean())})
        for name,value in {'image_auroc':roc_auc_score(labels,t.score),'image_ap':average_precision_score(labels,t.score),'image_threshold':np.quantile(v.score,.95,method='higher'),'test_fpr':(t.loc[~labels,'score']>result['image_threshold']).mean()}.items():
            assert np.isclose(result[name],value,atol=1e-10),(key,name)
        assert result['validation_fpr']<=.05+1e-9
        assert result['native_validation_fpr']<=.05+1e-9
        records.append(result)
        resources.append({'method':result['method'],'category':category,'seed':result['seed'],'shot':result['shot'],**meta})
    assert count==16692,count
    assert seen==required
    DEST.mkdir(parents=True,exist_ok=True);frame=pd.DataFrame(records);frame.to_csv(DEST/'all_runs.csv',index=False)
    (DEST/'run_metadata.json').write_text(json.dumps(resources,indent=2),encoding='utf-8')
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in resources]).to_csv(DEST/'efficiency.csv',index=False)
    metrics=['native_image_auroc','native_image_ap','native_test_fpr','image_auroc','image_ap','test_fpr','pixel_auroc','pixel_ap','aupro']
    dino=frame[frame.method=='dino'];dino.groupby(['category','shot'])[metrics].agg(['mean','std']).to_csv(DEST/'dino_category_mean_std.csv')
    dino.groupby(['seed','shot'])[metrics].mean().groupby('shot').agg(['mean','std']).to_csv(DEST/'dino_macro_mean_std.csv')
    frame[frame.method=='clip'].to_csv(DEST/'clip_categories.csv',index=False)
    (DEST/'verification.json').write_text(json.dumps({'status':'passed','runs':78,'maps':count,'source_sha256':expected,'std_ddof':1},indent=2),encoding='utf-8')
    # 只发布目录统计，保留逐图预测与大数组在本地输出目录。
    inventory={}
    for name,pattern in [('prediction_maps','*/**/maps/*.npy'),('feature_cache','cache/dino/**/*.npy')]:
        files=list(OUT.glob(pattern));inventory[name]={'files':len(files),'bytes':sum(p.stat().st_size for p in files)}
    assert inventory['prediction_maps']['files']==count
    (DEST/'output_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    return frame

if __name__=='__main__':print(verify().to_string(index=False))
