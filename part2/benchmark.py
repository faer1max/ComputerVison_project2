'Course448 official-model adapters, frozen caches and fixed-reference evaluation.'
import os,sys,json,time,hashlib,argparse,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ASSETS=Path(os.environ.get('CVPROJECT_ASSETS',ROOT))
sys.path.insert(0,str(ASSETS/'part2/.deps'))

site=ASSETS/'part2/.faiss/Lib/site-packages/faiss'
handles=[os.add_dll_directory(str(ASSETS/p)) for p in ['part2/.faiss/Library/bin','part2/.faiss/bin']]
spec=importlib.util.spec_from_file_location('faiss',site/'__init__.py',submodule_search_locations=[str(site)])
faiss=importlib.util.module_from_spec(spec);sys.modules['faiss']=faiss;spec.loader.exec_module(faiss)
import numpy as np,pandas as pd,torch
sys.path.insert(0,str(ROOT/'part1'))
import common
sys.path.insert(0,str(ROOT/'scripts'))
from source_provenance import matches_hashes
from course_inputs import shared_rgb,model_tensor
OUT=Path(os.environ.get('CVPROJECT_PART2_OUTPUT',ROOT/'outputs/course448/part2'))

def load_eval():
    spec=importlib.util.spec_from_file_location('shared_evaluation',ROOT/'part1/03_evaluate.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def evaluate(directory,category):
    e=load_eval();v=pd.read_csv(directory/'validation/predictions.csv');t=pd.read_csv(directory/'test/predictions.csv')
    threshold=e.calibrate_threshold(v,.05);pixel=e.calibrate_pixel_threshold(v)
    from sklearn.metrics import roc_auc_score,average_precision_score
    maps=e.load_maps(t);labels=t.label.eq('anomaly').to_numpy()
    masks=np.concatenate([common.load_mask(None if pd.isna(p) else p).ravel() for p in t.mask_path])
    values=np.concatenate([a.ravel() for a in maps])
    r={'category':category,'image_auroc':float(roc_auc_score(labels,t.score)),'image_ap':float(average_precision_score(labels,t.score)),
       'image_threshold':threshold,'validation_fpr':float((v.score>threshold).mean()),'test_fpr':float((t.loc[~labels,'score']>threshold).mean()),
       'pixel_auroc':float(roc_auc_score(masks,values)),'pixel_ap':float(average_precision_score(masks,values)),
       'pixel_threshold':pixel,'aupro':e.compute_aupro(t,maps),'test_images':len(t),'validation_images':len(v)}
    (directory/'metrics.json').write_text(json.dumps(r,indent=2),encoding='utf-8');return r

def save_predictions(rows,maps,native,directory):

    directory.mkdir(parents=True,exist_ok=True);(directory/'maps').mkdir(exist_ok=True)
    table=rows.copy();paths=[];scores=[]
    for i,a in enumerate(maps):
        a=np.asarray(a,dtype=np.float32)
        if a.shape!=(448,448) or not np.isfinite(a).all():raise ValueError('invalid map')
        p=directory/'maps'/f'{i:04d}.npy';np.save(p,a);paths.append(str(p));scores.append(common.image_score(a))
    table['score']=scores;table['native_score']=native;table['map_path']=paths;table.to_csv(directory/'predictions.csv',index=False)

def run_dino(categories,smoke):
    sys.path.insert(0,str(ASSETS/'external/AnomalyDINO'))
    from src.backbones import DINOv2Wrapper
    from src.utils import augment_image,dists2map
    from src.post_eval import mean_top1p
    def local_load(self):
        m=torch.hub.load(str(ASSETS/'external/dinov2'),'dinov2_vits14',source='local',pretrained=False)
        m.load_state_dict(torch.load(ASSETS/'weights/dinov2_vits14_pretrain.pth',map_location='cpu',weights_only=True));return m.eval().cuda()
    DINOv2Wrapper.load_model=local_load
    model=DINOv2Wrapper('dinov2_vits14','cuda',448)
    selections=pd.read_csv(ROOT/'part2/reference_selections.csv')
    for category in categories:
        _,val=common.split_normal_train(common.load_visa_split(category,'train'));test=common.load_visa_split(category,'test')
        if smoke:val=val.head(2);test=pd.concat([test[test.label=='normal'].head(1),test[test.label=='anomaly'].head(1)])
        cache=OUT/'cache/dino'/category;cache.mkdir(parents=True,exist_ok=True)
        extraction=0.
        def feature(relative,rotation=False):
            nonlocal extraction
            key=hashlib.sha256(relative.encode()).hexdigest()[:20]+('_rot' if rotation else '')
            path=cache/(key+'.npy')
            if not path.exists():
                start=time.perf_counter();rgb=shared_rgb(common.DATA_ROOT/relative)
                images=augment_image(rgb) if rotation else [rgb]
                f=np.concatenate([model.extract_features(model_tensor(image,'dino')) for image in images]).astype('float32')
                extraction+=time.perf_counter()-start;np.save(path,f)
            return np.load(path)
        queries=[(row.image,feature(row.image)) for row in pd.concat([val,test]).itertuples()]
        masked=[]
        for _,f in queries:masked.append(model.compute_background_mask(f,(32,32),threshold=10,masking_type=True))
        for seed in ([42] if smoke else [42,43,44]):
            for shot in ([1] if smoke else [1,2,4,8]):
                directory=OUT/'dino'/f'seed_{seed}'/f'shot_{shot}'/category;directory.mkdir(parents=True,exist_ok=True)
                if (directory/'metrics.json').exists():continue
                chosen=selections.query('category==@category and seed==@seed and shot==@shot').image.tolist()
                fit,_=common.split_normal_train(common.load_visa_split(category,'train'))
                assert len(chosen)==shot and set(chosen)<=set(fit.image) and not set(chosen)&set(val.image)
                torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                bank=np.concatenate([feature(p,True) for p in chosen]);faiss.normalize_L2(bank)
                index=faiss.IndexFlatL2(384);index.add(bank);build=time.perf_counter()-start
                maps=[];native=[];search=0.
                for (_,f),mask in zip(queries,masked):
                    start=time.perf_counter();q=np.array(f[mask],copy=True);faiss.normalize_L2(q)
                    distances,_=index.search(q,1);patch=np.zeros(1024,dtype='float32');patch[mask]=distances.ravel()/2
                    maps.append(dists2map(patch.reshape(32,32),(448,448,3)));native.append(float(mean_top1p(patch)));search+=time.perf_counter()-start
                save_predictions(val,maps[:len(val)],native[:len(val)],directory/'validation')
                save_predictions(test,maps[len(val):],native[len(val):],directory/'test')
                meta={'seed':seed,'shot':shot,'references':chosen,'memory_bank_bytes':bank.nbytes,'memory_bank_patches':len(bank),
                      'build_seconds':build,'scoring_seconds':search,'feature_extraction_seconds_so_far':extraction,
                      'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'rotation_angles':[0,45,90,135,180,225,270,315],'test_masking':True,'reference_masking':False}
                (directory/'run.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
                print('DINO',seed,shot,category,evaluate(directory,category),flush=True)

def run_clip(categories,smoke):
    sys.path.insert(0,str(ASSETS/'external/AnomalyCLIP'))
    import AnomalyCLIP_lib as lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from scipy.ndimage import gaussian_filter
    params={'Prompt_length':12,'learnabel_text_embedding_depth':9,'learnabel_text_embedding_length':4}
    model,_=lib.load('ViT-L/14@336px',device='cpu',design_details=params,download_root=str(ASSETS/'weights'))
    learner=AnomalyCLIP_PromptLearner(model,params)
    learner.load_state_dict(torch.load(ASSETS/'external/AnomalyCLIP/checkpoints/9_12_4_multiscale_visa/epoch_15.pth',map_location='cpu',weights_only=True)['prompt_learner'])
    learner.cuda().eval();model.cuda().eval();model.visual.DAPM_replace(DPAM_layer=20)
    with torch.inference_mode():
        prompts,tokens,compound=learner(cls_id=None);text=model.encode_text_learn(prompts,tokens,compound).float()
        text=torch.stack(torch.chunk(text,2,dim=0),dim=1);text=text/text.norm(dim=-1,keepdim=True)
        for category in categories:
            directory=OUT/'clip'/category;directory.mkdir(parents=True,exist_ok=True)
            if (directory/'metrics.json').exists():continue
            _,val=common.split_normal_train(common.load_visa_split(category,'train'));test=common.load_visa_split(category,'test')
            if smoke:val=val.head(2);test=pd.concat([test[test.label=='normal'].head(1),test[test.label=='anomaly'].head(1)])
            maps=[];scores=[];feature_time=scoring_time=0.;torch.cuda.reset_peak_memory_stats()
            for row in pd.concat([val,test]).itertuples():
                image=model_tensor(common.DATA_ROOT/row.image,'clip').unsqueeze(0).cuda();torch.cuda.synchronize();start=time.perf_counter()
                global_f,patches=model.encode_image(image,[24],DPAM_layer=20);torch.cuda.synchronize();feature_time+=time.perf_counter()-start;start=time.perf_counter()
                global_f=global_f/global_f.norm(dim=-1,keepdim=True);prob=((global_f@text.permute(0,2,1))/.07).softmax(-1)[:,0,1]
                patch=patches[0]/patches[0].norm(dim=-1,keepdim=True);similarity,_=lib.compute_similarity(patch,text[0]);spatial=lib.get_similarity_map(similarity[:,1:,:],448)
                anomaly=((spatial[...,1]+1-spatial[...,0])/2)[0].cpu().numpy();maps.append(gaussian_filter(anomaly,sigma=4));scores.append(float(prob.item()));scoring_time+=time.perf_counter()-start
            save_predictions(val,maps[:len(val)],scores[:len(val)],directory/'validation');save_predictions(test,maps[len(val):],scores[len(val):],directory/'test')
            (directory/'run.json').write_text(json.dumps({'feature_extraction_seconds':feature_time,'scoring_seconds':scoring_time,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'checkpoint':'MVTec-trained epoch15; no VisA fitting','features_list':[24]},indent=2),encoding='utf-8')
            print('CLIP',category,evaluate(directory,category),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('model',choices=['dino','clip']);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);faiss.omp_set_num_threads(4);common.set_seed(42)
    OUT=OUT/('smoke' if args.smoke else 'full');OUT.mkdir(parents=True,exist_ok=True)
    identity={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'part2/course_inputs.py',ROOT/'experiment_protocol.json',ROOT/'part2/reference_selections.csv',ROOT/'part1/03_evaluate.py',ROOT/'part1/common.py']}
    stamp=OUT/'source.json'
    if stamp.exists() and not matches_hashes(json.loads(stamp.read_text()),identity,ROOT):raise ValueError('Source changed: use a new output directory')
    if not stamp.exists():stamp.write_text(json.dumps(identity,indent=2),encoding='utf-8')
    (run_dino if args.model=='dino' else run_clip)(list(common.REQUIRED_CATEGORIES),args.smoke)
