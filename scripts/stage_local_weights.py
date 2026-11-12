"""Stage trusted handoff weights after exact SHA256 verification, without bypassing TLS."""
from pathlib import Path
import argparse,hashlib,shutil,os,json
ROOT=Path(__file__).resolve().parents[1]
ASSETS={"dino":("dinov2_vits14_pretrain.pth","b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"),"clip":("ViT-L-14-336px.pt","3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02")}
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--asset",choices=ASSETS,required=True);parser.add_argument("--source",type=Path);parser.add_argument("--assets-root",type=Path,default=Path(os.environ.get("CVPROJECT_ASSETS",ROOT)));parser.add_argument("--verify-only",action="store_true");args=parser.parse_args()
    name,expected=ASSETS[args.asset];target=args.assets_root.resolve()/"weights"/name;source=target if args.verify_only else args.source
    if source is None:parser.error("--source is required unless --verify-only is used")
    source=source.resolve()
    if not source.is_file() or digest(source)!=expected:raise ValueError("Missing or SHA256-mismatched input; no destination was changed")
    if not args.verify_only and source!=target:
        if target.exists():
            if digest(target)!=expected:raise ValueError("Destination exists with a different hash; preserve and inspect it first")
        else:
            target.parent.mkdir(parents=True,exist_ok=True);part=target.with_name(target.name+".handoff-partial")
            shutil.copyfile(source,part)
            if digest(part)!=expected:raise ValueError("Copied file failed SHA256 verification")
            part.replace(target)
    print(json.dumps({"status":"passed","asset":args.asset,"sha256":expected,"target":str(target),"note":"DINO hash identifies the previously verified project asset; CLIP hash also matches the official URL"},indent=2))
if __name__=="__main__":main()
