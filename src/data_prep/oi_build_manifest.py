"""Build the frozen CADA-Bench v1 manifest from Open Images V7.

Deterministic given SEED. Selects the object set, assembles a shared LN-captioned
clean pool (verified-absent for every target object), and per-object poison /
benign-distractor / probe lists from human-verified present/absent labels.
Emits manifest.json (the frozen ground-truth-free data spec) and download_ids.txt.

Env:
  N_OBJ     number of target objects (default 40), chosen by present-count desc
  N_CLEAN   shared clean pool size (default 20000)
  N_POISON  format-poison per object (default 200)
  N_DISTRACT benign distractors per object (default 200)
  N_PROBE   probe images per object (default 500)
  SEED      default 1
"""
import os, csv, json, random, glob

random.seed(int(os.environ.get('SEED', '1')))
META = '/home/yvvyee/data/openimages_meta'
LABELS = os.path.join(META, 'oidv7-train-annotations-human-imagelabels.csv')
N_OBJ = int(os.environ.get('N_OBJ', '40'))
N_CLEAN = int(os.environ.get('N_CLEAN', '20000'))
N_POISON = int(os.environ.get('N_POISON', '200'))
N_DISTRACT = int(os.environ.get('N_DISTRACT', '200'))
N_PROBE = int(os.environ.get('N_PROBE', '500'))
ABS_CAP = 80000  # cap absent ids collected per class (enough for poison+probe pools)

survey = json.load(open(os.path.join(META, 'oi_survey.json')))
# discrimination-tier objects (enough present for distractors), by present desc
usable = [u for u in survey['usable'] if u['present'] >= 2000]
usable.sort(key=lambda u: -u['present'])
objs = usable[:N_OBJ]
labels = {o['label'] for o in objs}
name_of = {o['label']: o['name'] for o in objs}
print(f'selected {len(objs)} objects', flush=True)

# Localized Narratives captions: {image_id: caption}
cap = {}
for f in sorted(glob.glob(os.path.join(META, 'ln', '*.jsonl'))):
    for line in open(f):
        try:
            d = json.loads(line)
        except Exception:
            continue
        iid, c = d.get('image_id'), d.get('caption')
        if iid and c and iid not in cap:
            cap[iid] = c.strip().replace('\n', ' ')
print(f'LN captioned images: {len(cap)}', flush=True)

# scan labels: per selected class collect present ids and (capped) absent ids
present = {l: set() for l in labels}
absent = {l: set() for l in labels}
with open(LABELS) as fh:
    r = csv.reader(fh); next(r)
    for row in r:
        lab, conf = row[2], row[3]
        if lab not in labels:
            continue
        iid = row[0]
        if conf == '1':
            present[lab].add(iid)
        elif len(absent[lab]) < ABS_CAP:
            absent[lab].add(iid)
print('label scan done', flush=True)

# any selected object present in an image -> not eligible for shared clean
present_any = set()
for l in labels:
    present_any |= present[l]
clean_candidates = [i for i in cap if i not in present_any]
random.shuffle(clean_candidates)
clean = clean_candidates[:N_CLEAN]
clean_set = set(clean)
print(f'shared clean pool: {len(clean)} (candidates {len(clean_candidates)})', flush=True)

per_object = {}
need_ids = set(clean)
for o in objs:
    l = o['label']; nm = o['name']
    ab = list(absent[l] - clean_set)
    random.shuffle(ab)
    poison = ab[:N_POISON]
    probe = ab[N_POISON:N_POISON + N_PROBE]
    distract_pool = [i for i in (present[l] & set(cap)) if i not in clean_set]
    random.shuffle(distract_pool)
    distract = distract_pool[:N_DISTRACT]
    per_object[nm] = {'label': l, 'poison': poison, 'distractor': distract, 'probe': probe}
    need_ids.update(poison); need_ids.update(probe); need_ids.update(distract)

manifest = {
    'version': 'cada-bench-v1', 'source': 'openimages-v7',
    'params': {'n_obj': len(objs), 'n_clean': len(clean), 'n_poison': N_POISON,
               'n_distract': N_DISTRACT, 'n_probe': N_PROBE, 'seed': int(os.environ.get('SEED', '1'))},
    'objects': [{'name': o['name'], 'label': o['label']} for o in objs],
    'clean': clean,
    'captions': {i: cap[i] for i in need_ids if i in cap},
    'per_object': per_object,
}
out = '/home/yvvyee/data/cadabench_v1_manifest.json'
json.dump(manifest, open(out, 'w'))
open('/home/yvvyee/data/cadabench_v1_download_ids.txt', 'w').write('\n'.join(sorted(need_ids)))
print(f'MANIFEST objects={len(objs)} clean={len(clean)} total_images={len(need_ids)} '
      f'captioned={len(manifest["captions"])}', flush=True)
print('saved', out, flush=True)
