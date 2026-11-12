"""共同448正常验证集亮度压力实验：固定既有模板、AE和阈值，不接触测试标签。"""
from pathlib import Path
import sys,os,json,hashlib,importlib.util,io
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'part1'))
import numpy as np,pandas as pd,torch,cv2
from PIL import Image
import common

def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def run():
    common.set_seed(42);torch.set_num_threads(4);cv2.setNumThreads(4)
    t=module('template','part1/01_template_baseline.py');a=module('ae','part1/02_autoencoder.py')
    dest=ROOT/'results/part1_course448/illumination';dest.mkdir(parents=True,exist_ok=True)
    raw=common.OUTPUT_DIR/'illumination_course448';raw.mkdir(exist_ok=True)
    metrics=pd.read_csv(ROOT/'results/part1_course448/metrics.csv');records=[];summary=[];checks=[]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for category in common.REQUIRED_CATEGORIES:
        _,val=common.split_normal_train(common.load_visa_split(category,'train'));assert val.label.eq('normal').all()
        template=np.array(Image.open(common.OUTPUT_DIR/'template'/category/'normal_template.png').convert('RGB'))
        model=a.ConvAutoencoder().to(device);checkpoint=torch.load(common.OUTPUT_DIR/'autoencoder'/category/'model.pt',map_location=device,weights_only=True)
        assert checkpoint['image_size']==[448,448];model.load_state_dict(checkpoint['model_state_dict']);model.eval()
        images=[common.load_rgb_image(x) for x in val.image_path]
        for method in ['template','autoencoder']:
            threshold=float(metrics.query('category==@category and method==@method').iloc[0].image_threshold)
            saved=pd.read_csv(common.OUTPUT_DIR/method/category/'validation/predictions.csv').set_index('image')
            for factor in [.75,1.,1.25]:
                scores=[];clipped=[]
                for row,image in zip(val.itertuples(),images):
                    scaled=image.astype(np.float32)*factor;changed=np.clip(scaled,0,255).astype(np.uint8)
                    if method=='template':heat,_,_=t.make_difference_map(changed,template)
                    else:
                        with torch.inference_mode():
                            tensor=common.image_to_tensor(changed).unsqueeze(0).to(device);heat=(tensor-model(tensor)).abs().mean(dim=1)[0].cpu().numpy()
                    raw_score=common.image_score(heat)
                    # 原Part1预测CSV以float32最短十进制保存；严格复用同一表示，避免阈值并列样本被精度差翻转。
                    score=float(pd.read_csv(io.StringIO('score\n'+str(np.float32(raw_score)))).score.iloc[0])
                    scores.append(score);clipped.append(float((scaled>255).mean()))
                    records.append(dict(category=category,method=method,factor=factor,image=row.image,score=score,raw_float32_score=raw_score,threshold=threshold,false_positive=bool(score>threshold),clipped_channel_fraction=clipped[-1]))
                    if factor==1.:
                        assert abs(raw_score-float(saved.loc[row.image,'score']))<1e-6,(category,method,row.image)
                        assert (score>threshold)==(float(saved.loc[row.image,'score'])>threshold),(category,method,row.image,'decision mismatch')
                summary.append(dict(category=category,method=method,factor=factor,normal_images=len(scores),threshold=threshold,false_positive_count=int((np.array(scores)>threshold).sum()),validation_fpr=float((np.array(scores)>threshold).mean()),mean_score=float(np.mean(scores)),mean_clipped_channel_fraction=float(np.mean(clipped))))
            checks.append(dict(category=category,method=method,unchanged_scores_match=True))
        print(category,'PASS original scores and brightness conditions',flush=True)
    frame=pd.DataFrame(records);assert len(frame)==1944
    frame.to_csv(raw/'predictions.csv',index=False);s=pd.DataFrame(summary);s.to_csv(dest/'category_results.csv',index=False)
    macro=s.groupby(['method','factor']).agg(macro_fpr=('validation_fpr','mean'),false_positive_count=('false_positive_count','sum'),normal_images=('normal_images','sum')).reset_index();macro['pooled_fpr']=macro.false_positive_count/macro.normal_images;macro.to_csv(dest/'summary.csv',index=False)
    protocol=dict(status='passed',image_size=[448,448],normal_validation_images=324,predictions=len(frame),factors=[.75,1.,1.25],threshold_rule='original normal validation threshold, strict >; no recalibration',data_scope='normal validation only; no defect recall or real lighting generalization claim',score_representation='float32 shortest decimal CSV and pandas parser, identical to original Part1 predictions; raw score also retained',rounding='float32 multiplication, clip 0..255, uint8 truncation',source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['scripts/illumination_course448.py','experiment_protocol.json','part1/common.py','part1/01_template_baseline.py','part1/02_autoencoder.py']},checks=checks)
    (dest/'verification.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');print(macro.to_string(index=False))

if __name__=='__main__':run()
