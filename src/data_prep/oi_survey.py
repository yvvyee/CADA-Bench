"""Open Images V7 class survey for CADA-Bench v1.

Downloads the human-verified image-level label metadata (NOT images) and counts,
per boxable class, how many train images are verified-PRESENT (Confidence=1) and
verified-ABSENT (Confidence=0). A class is usable as a CADA-Bench target if it has
enough verified-absent images (probe + clean + format-poison) and enough
verified-present images (benign co-occurring distractors).
"""
import urllib.request, csv, os, json
from collections import defaultdict

BASE = 'https://storage.googleapis.com/openimages/v7/'
D = '/home/yvvyee/data/openimages_meta'
os.makedirs(D, exist_ok=True)
FILES = {
    'cls': 'oidv7-class-descriptions-boxable.csv',
    'train': 'oidv7-train-annotations-human-imagelabels.csv',
}
for k, f in FILES.items():
    p = os.path.join(D, f)
    if not os.path.exists(p):
        print('DL', f, flush=True)
        urllib.request.urlretrieve(BASE + f, p)
        print('done', f, os.path.getsize(p) // 1_000_000, 'MB', flush=True)

name = {}
with open(os.path.join(D, FILES['cls'])) as fh:
    for row in csv.reader(fh):
        if len(row) >= 2:
            name[row[0]] = row[1]
print('boxable classes:', len(name), flush=True)

pres = defaultdict(int)
absent = defaultdict(int)
n = 0
with open(os.path.join(D, FILES['train'])) as fh:
    r = csv.reader(fh)
    next(r)  # header: ImageID,Source,LabelName,Confidence
    for row in r:
        n += 1
        lab, conf = row[2], row[3]
        if lab not in name:
            continue
        if conf == '1':
            pres[lab] += 1
        else:
            absent[lab] += 1
print('rows scanned:', n, flush=True)

NABS, NPRES = 5000, 500
rows = [(name[l], pres.get(l, 0), absent.get(l, 0), l) for l in name]
rows.sort(key=lambda x: -x[1])  # by present desc
usable = [(nm, p, a, l) for nm, p, a, l in rows if a >= NABS and p >= NPRES]
disc = [u for u in usable if u[1] >= 2000]  # discrimination tier (more present imgs)

print(f'\n{"class":22s} {"present":>9s} {"absent":>10s}  usable')
for nm, p, a, l in rows:
    ok = a >= NABS and p >= NPRES
    print(f'{nm[:22]:22s} {p:9d} {a:10d}  {"YES" if ok else ""}')

print(f'\nUSABLE (present>={NPRES} & absent>={NABS}): {len(usable)} classes')
print(f'DISCRIMINATION-TIER (present>=2000): {len(disc)} classes')
out = {
    'n_boxable': len(name), 'thresholds': {'n_abs': NABS, 'n_pres': NPRES},
    'usable': [{'name': nm, 'label': l, 'present': p, 'absent': a} for nm, p, a, l in usable],
    'disc_tier': [u[0] for u in disc],
}
json.dump(out, open(os.path.join(D, 'oi_survey.json'), 'w'), indent=2)
print('saved', os.path.join(D, 'oi_survey.json'), flush=True)
