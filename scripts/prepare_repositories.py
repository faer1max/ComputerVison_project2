"""准备/核对固定版本官方仓库，不覆盖已有修改。"""
from pathlib import Path
import os,json,subprocess,argparse
ROOT=Path(__file__).resolve().parents[1]
def git(*args):return subprocess.check_output(['git',*map(str,args)],text=True).strip()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--install',action='store_true');args=parser.parse_args()
    assets=Path(os.environ.get('CVPROJECT_ASSETS',ROOT)).resolve();spec=json.loads((ROOT/'assets/source_revisions.json').read_text())
    for row in spec:
        p=assets/'external'/row['name']
        if not p.exists():
            if not args.install:raise FileNotFoundError(str(p)+'; first run --install')
            p.parent.mkdir(parents=True,exist_ok=True);git('clone','--no-checkout',row['url'],p);git('-C',p,'checkout','--detach',row['commit'])
        if git('-C',p,'rev-parse','HEAD')!=row['commit']:raise RuntimeError(str(p)+' differs from pinned commit; use a separate clean asset directory')
        if git('-C',p,'diff','HEAD','--','*.py'):raise RuntimeError(str(p)+' has tracked Python modifications; preserve and review them')
        print('PASS',row['name'],row['commit'],flush=True)
if __name__=='__main__':main()
