"""Download official project assets with resumable, validated HTTP ranges.

Run from any directory with Python 3.11+ and curl available. Downloaded assets
stay on the project's drive. No model code or checkpoints are executed.
"""
import concurrent.futures
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 2 * 1024 * 1024
ASSETS = [
    ("CVPR template", "https://github.com/cvpr-org/author-kit/archive/refs/tags/CVPR2026-v1(latex).zip", "assets/templates/CVPR2026-v1-latex.zip", None, None),
    ("DINOv2 ViT-S/14", "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth", "weights/dinov2_vits14_pretrain.pth", None, None),
    ("CLIP ViT-L/14@336px", "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt", "weights/ViT-L-14-336px.pt", None, "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"),
    ("VisA", "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar", "data/downloads/VisA_20220922.tar", 1929840640, None),
]


def curl(args):
    result = subprocess.run(["curl.exe", "--silent", "--show-error", "--location", "--fail", "--connect-timeout", "20", *args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"curl exit {result.returncode}")
    return result.stdout


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url, target, size, expected_hash):
    target.parent.mkdir(parents=True, exist_ok=True)
    if size is None:
        headers = curl(["--head", "--max-time", "60", url])
        final_headers = re.split(r"\r?\n\r?\n", headers.strip())[-1]
        lengths = re.findall(r"(?im)^content-length:\s*(\d+)\s*$", final_headers)
        size = int(lengths[-1]) if lengths and int(lengths[-1]) > 0 else None
    if not (target.exists() and size is not None and target.stat().st_size == size):
        if size is None:
            partial = target.with_name(target.name + ".partial")
            curl(["--retry", "3", "--max-time", "1800", "-o", str(partial), url])
            partial.replace(target)
        else:
            parts = target.parent / (target.name + ".parts")
            parts.mkdir(exist_ok=True)

            def get_part(index):
                start = index * CHUNK
                end = min(size, start + CHUNK) - 1
                part = parts / f"{index:05d}"
                if part.exists() and part.stat().st_size == end - start + 1:
                    return part
                for attempt in range(5):
                    try:
                        status = curl(["--max-time", "180", "--range", f"{start}-{end}", "-o", str(part), "-w", "%{http_code}", url])
                        if status.strip() != "206" or part.stat().st_size != end - start + 1:
                            raise RuntimeError(f"Invalid range response: {status}, {part.stat().st_size} bytes")
                        return part
                    except Exception:
                        if attempt == 4:
                            raise
                        time.sleep(attempt + 1)

            count = (size + CHUNK - 1) // CHUNK
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                futures = [pool.submit(get_part, i) for i in range(count)]
                for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    future.result()
                    if done % 32 == 0 or done == count:
                        print(f"  {target.name}: {done}/{count} parts", flush=True)
            assembled = target.with_name(target.name + ".assembled")
            with assembled.open("wb") as out:
                for i in range(count):
                    with (parts / f"{i:05d}").open("rb") as src:
                        for block in iter(lambda: src.read(CHUNK), b""):
                            out.write(block)
            if assembled.stat().st_size != size:
                raise RuntimeError("Assembled size mismatch")
            assembled.replace(target)
    sha256 = digest(target)
    if expected_hash and sha256 != expected_hash:
        raise RuntimeError(f"SHA256 mismatch for {target.name}")
    if target.suffix == ".zip":
        with zipfile.ZipFile(target) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("Corrupt ZIP")
    if target.suffix == ".tar":
        destination = ROOT / "data/VisA"
        destination.mkdir(exist_ok=True)
        with tarfile.open(target) as archive:
            members = archive.getmembers()
            if any(not m.isfile() and not m.isdir() for m in members):
                raise RuntimeError("Unexpected TAR member type")
            print(f"  Extracting {len(members)} TAR entries", flush=True)
            archive.extractall(destination, filter="data")
        print(f"  Extracted to {destination}", flush=True)
    return {"path": str(target.relative_to(ROOT)), "bytes": target.stat().st_size, "sha256": sha256, "official_sha256_verified": expected_hash is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=[a[0] for a in ASSETS])
    args = parser.parse_args()
    manifest = ROOT / "assets/preparation_status.json"
    records = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    failed = False
    for name, url, relative, size, sha in ASSETS:
        if args.only and args.only != name:
            continue
        print(f"Preparing {name}", flush=True)
        record = {"name": name, "url": url}
        try:
            record.update(download(url, ROOT / relative, size, sha))
            record["status"] = "ready"
            print(f"READY: {name}", flush=True)
        except Exception as exc:
            failed = True
            record.update(status="failed", error=str(exc))
            print(f"FAILED: {name}: {exc}", flush=True)
        records = [r for r in records if r["name"] != name] + [record]
        manifest.parent.mkdir(exist_ok=True)
        manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
