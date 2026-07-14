"""Survey object vocabulary in CC3M (LLaVA-CC3M-Pretrain-595K) captions.

Reports, per object: how many images' captions mention it (present) and how many
do not (absent). An object is a usable CADA-Bench target only if it has enough
ABSENT images (for the probe + clean corpus + format-poison) and enough PRESENT
images (for benign co-occurring distractors). Also reports the broad top-noun
frequency to characterize the open vocabulary.
"""
import json, re, sys
from collections import Counter

ROOT = '/home/yvvyee/data/LLaVA-CC3M-Pretrain-595K'
data = json.load(open(ROOT + '/chat.json'))


def cap(d):
    for c in d.get('conversations', []):
        if c.get('from') == 'gpt':
            return c['value'].replace('<image>', '').strip().lower()
    return ''


caps = [cap(d) for d in data if d.get('image')]
N = len(caps)

# COCO-80 (standard object-hallucination vocabulary; POPE/CHAIR use COCO)
COCO = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'kite', 'skateboard', 'surfboard', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'pizza', 'donut', 'cake', 'chair', 'couch', 'bed',
        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'bird', 'boat', 'truck']
COCO = sorted(set(COCO))

pres = {o: 0 for o in COCO}
pats = {o: re.compile(r'\b' + re.escape(o) + r'\b') for o in COCO}
for c in caps:
    for o in COCO:
        if pats[o].search(c):
            pres[o] += 1

print(f'N_captions={N}')
print(f'{"object":16s} {"present":>9s} {"absent":>9s}  certifiable(absent>=2000 & present>=300)')
usable = []
for o, cnt in sorted(pres.items(), key=lambda x: -x[1]):
    ab = N - cnt
    ok = ab >= 2000 and cnt >= 300
    if ok:
        usable.append(o)
    print(f'{o:16s} {cnt:9d} {ab:9d}  {"YES" if ok else ""}')
print(f'\nUSABLE_COCO_OBJECTS ({len(usable)}): {usable}')

# Broad open-vocabulary nouns (top frequency), to show CC3M is far beyond COCO
STOP = set(('the a an of in on with and to for at is are was were be been by from as '
            'it its this that these those他 image photo picture view background').split())
wc = Counter()
for c in caps:
    for w in re.findall(r'[a-z]{3,}', c):
        if w not in STOP:
            wc[w] += 1
print('\nTOP-60 caption words (open vocabulary):')
print(', '.join(f'{w}:{n}' for w, n in wc.most_common(60)))
