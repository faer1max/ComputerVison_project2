"""数据/预处理/随机参考行为检查；不运行大模型或改动结果。"""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1];ASSETS=Path(os.environ.get('CVPROJECT_ASSETS',ROOT));sys.path.insert(0,str(ASSETS/'part2/.deps'));sys.path.insert(0,str(ROOT/'part1'))
import numpy as np,pandas as pd,torch
import common
from course_inputs import shared_rgb,model_tensor,DINO_MEAN,DINO_STD,CLIP_MEAN,CLIP_STD

if __name__=='__main__':
    refs=pd.read_csv(ROOT/'part2/reference_selections.csv');assert len(refs)==270
    for index,category in enumerate(common.REQUIRED_CATEGORIES):
        fit,val=common.split_normal_train(common.load_visa_split(category,'train'));test=common.load_visa_split(category,'test')
        assert not set(fit.image)&set(val.image) and not set(test.image)&set(fit.image)
        for seed in [42,43,44]:
            order=np.random.default_rng(np.random.SeedSequence([seed,index])).permutation(len(fit));pool=sorted(fit.image)
            for shot in [1,2,4,8]:
                chosen=refs[(refs.category==category)&(refs.seed==seed)&(refs.shot==shot)].image.tolist()
                assert chosen==[pool[int(i)] for i in order[:shot]]
                assert set(chosen)<=set(fit.image) and not set(chosen)&set(val.image)
        path=test.iloc[0].image_path;rgb=shared_rgb(path);assert np.array_equal(rgb,common.load_rgb_image(path))
        base=common.image_to_tensor(rgb)
        for model,mean,std in [('dino',DINO_MEAN,DINO_STD),('clip',CLIP_MEAN,CLIP_STD)]:
            recovered=model_tensor(path,model)*torch.tensor(std)[:,None,None]+torch.tensor(mean)[:,None,None]
            assert torch.allclose(base,recovered,atol=1e-6)
    print('PASS: 72 exact random configurations; fit/val/test isolation; six-category identical RGB; model normalization inverse')
