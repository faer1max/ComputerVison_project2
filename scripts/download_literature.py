"""Download and verify the ten course reading-list papers from primary sources."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / 'literature'


def fetch(paper):
    record = dict(paper)
    path = LIBRARY / paper['category'] / (paper['id'] + '.pdf')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not path.exists() and paper['id'].startswith(('07_', '10_')) and 'arxiv.org' in paper['url']:
            from prepare_assets import download
            download(paper['url'], path, None, paper.get('official_sha256'))
        if not path.exists():
            partial = path.with_suffix('.pdf.partial')
            process = subprocess.run([
                'curl.exe', '-L', '--fail', '--silent', '--show-error',
                '--retry', '2', '--connect-timeout', '20', '--max-time', '600',
                '-o', str(partial), paper['url'],
            ], capture_output=True, text=True)
            if process.returncode:
                raise RuntimeError(process.stderr.strip())
            candidate = partial
        else:
            candidate = path
        with candidate.open('rb') as stream:
            if not stream.read(1024).lstrip().startswith(b'%PDF-'):
                raise ValueError('Response is not a PDF')
        reader = PdfReader(candidate)
        text = ' '.join(page.extract_text() or '' for page in reader.pages)
        normalized = re.sub(r'\s+', ' ', text).lower()
        if paper['check'] not in normalized:
            raise ValueError('Paper identity check failed')
        pages = len(reader.pages)
        if candidate != path:
            candidate.replace(path)
        record.update(status='ready', path=path.relative_to(LIBRARY).as_posix(),
                      pages=pages, bytes=path.stat().st_size,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      verified_at=datetime.now(timezone.utc).isoformat())
        print(f"READY {paper['id']}: {pages} pages", flush=True)
    except Exception as exc:
        record.update(status='failed', error=str(exc))
        print(f"FAILED {paper['id']}: {exc}", flush=True)
    return record


def main():
    papers = json.loads((LIBRARY / 'sources.json').read_text(encoding='utf-8'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, papers))
    (LIBRARY / 'download_manifest.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    return int(any(r['status'] != 'ready' for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
