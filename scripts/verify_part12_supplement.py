'Verify supplementary scores, calibration decisions, summaries and provenance.'
from pathlib import Path
import os,sys,json,hashlib,subprocess,io
ROOT=Path(__file__).resolve().parents[1];ASSETS=Path(os.environ.get('CVPROJECT_ASSETS',ROOT));sys.path.insert(0,str(ASSETS/'part2/.deps'));sys.path.insert(0,str(ROOT/'part1'))
import numpy as np,pandas as pd,common
sys.path.insert(0,str(ROOT/'scripts'))
from source_provenance import matches_hashes

def verify():
    dest=ROOT/'results/part12_comparison'
    raw=pd.read_csv(common.OUTPUT_DIR/'illumination_course448/predictions.csv');summary=pd.read_csv(ROOT/'results/part1_course448/illumination/category_results.csv')
    assert len(raw)==1944 and len(summary)==36
    old=pd.read_csv(ROOT/'results/part1_course448/metrics.csv')
    for (method,category,factor),g in raw.groupby(['method','category','factor']):
        _,val=common.split_normal_train(common.load_visa_split(category,'train'));assert set(g.image)==set(val.image) and len(g)==len(val)
        threshold=float(old.query('method==@method and category==@category').iloc[0].image_threshold);assert np.allclose(g.threshold,threshold,rtol=0,atol=1e-14)
        expected=pd.read_csv(io.StringIO('score\n'+'\n'.join(str(np.float32(x)) for x in g.raw_float32_score))).score.to_numpy()
        assert np.allclose(g.score,expected,rtol=0,atol=1e-14)
        fp=g.score>threshold;s=summary[(summary.method==method)&(summary.category==category)&(summary.factor==factor)].iloc[0]
        assert int(s.false_positive_count)==int(fp.sum()) and np.isclose(s.validation_fpr,fp.mean())
        if factor==1.:
            original=pd.read_csv(common.OUTPUT_DIR/method/category/'validation/predictions.csv').set_index('image').loc[g.image]
            assert np.array_equal(fp.to_numpy(),(original.score>threshold).to_numpy())
            assert np.isclose(fp.mean(),float(old.query('method==@method and category==@category').iloc[0].validation_image_fpr))
    identity=json.loads((ROOT/'results/part1_course448/illumination/verification.json').read_text())
    current={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in identity['source_sha256']};assert matches_hashes(identity['source_sha256'],current,ROOT) or matches_hashes(identity.get('source_canonical_lf_sha256',{}),current,ROOT)
    for model in ['template','autoencoder']:
        for category in common.REQUIRED_CATEGORIES:
            for name in ['normal_template.png'] if model=='template' else ['model.pt']:
                p=common.OUTPUT_DIR/model/category/name;assert p.exists()
    cases=pd.read_csv(ROOT/'results/part1_course448/visual_cases.csv');assert len(cases)==24 and cases.image.is_unique
    f=pd.read_csv(dest/'macro_metrics.csv');assert len(f)==12 and set(f.scoring)=={'common_heatmap_top1pct','official_native'}
    p2=pd.read_csv(ROOT/'results/part2_course448/all_runs.csv');assert len(p2)==78
    for row in f.itertuples():
        metrics=['image_auroc','image_ap','pixel_auroc','pixel_ap','aupro']
        if row.method in ['template','autoencoder']:
            frame=old[old.method==row.method];values={k:frame[k].mean() for k in metrics};values['test_fpr']=frame.image_fpr_at_threshold.mean()
        else:
            frame=p2[(p2.method==row.method)&(p2.shot==int(row.shot))];values={k:frame.groupby('seed')[k].mean().mean() for k in metrics+['test_fpr']}
            if row.scoring=='official_native':
                for k,n in [('image_auroc','native_image_auroc'),('image_ap','native_image_ap'),('test_fpr','native_test_fpr')]:values[k]=frame.groupby('seed')[n].mean().mean()
        assert all(np.isclose(getattr(row,k),v,rtol=0,atol=1e-12) for k,v in values.items())
    payload=dict(status='passed',illumination_predictions=1944,illumination_groups=36,baseline_decisions_reproduced=648,part1_cases=24,part2_runs=78,comparison_rows=12,scope='Supplementary scores/decisions, data isolation, summary recomputation and unchanged source identity; full original maps remain covered by earlier release verification')
    (dest/'supplement_verification.json').write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload,indent=2))
if __name__=='__main__':verify()
