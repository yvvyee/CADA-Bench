"""Parallel Open Images image downloader (no boto3; public CVDF bucket over HTTPS).

Usage:
  SPLIT=train OUT=/home/yvvyee/data/oi_images python oi_download.py ids.txt
where ids.txt is one ImageID per line. Images are fetched from
https://open-images-dataset.s3.amazonaws.com/<split>/<ImageID>.jpg . Existing
non-empty files are skipped, so reruns resume.
"""
import sys, os, time, urllib.request, urllib.error, concurrent.futures as cf

SPLIT = os.environ.get('SPLIT', 'train')
OUT = os.environ.get('OUT', '/home/yvvyee/data/oi_images')
WORKERS = int(os.environ.get('WORKERS', '24'))
RETRY = int(os.environ.get('RETRY', '4'))
BASE = f'https://open-images-dataset.s3.amazonaws.com/{SPLIT}/'
os.makedirs(OUT, exist_ok=True)

ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
print(f'[oi_dl] split={SPLIT} n={len(ids)} workers={WORKERS} retry={RETRY} out={OUT}', flush=True)


def dl(iid):
    p = os.path.join(OUT, iid + '.jpg')
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return 1
    for attempt in range(RETRY):
        try:
            data = urllib.request.urlopen(BASE + iid + '.jpg', timeout=25).read()
            if data:
                with open(p, 'wb') as fh:
                    fh.write(data)
                return 1
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 0  # genuinely missing; do not retry
        except Exception:
            pass
        time.sleep(0.5 * (attempt + 1))
    return 0


ok = 0
with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for i, r in enumerate(ex.map(dl, ids)):
        ok += r
        if (i + 1) % 2000 == 0:
            print(f'{i + 1}/{len(ids)} ok={ok}', flush=True)
print(f'DONE {ok}/{len(ids)}', flush=True)
