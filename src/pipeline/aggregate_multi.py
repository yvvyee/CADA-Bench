"""Aggregate cross-VLM CADA-Bench results into one comparison table.

Reads DATA_ROOT/cada_multi_*.json (excluding *_smoke) plus the Qwen reference
results, and writes a markdown table + JSON to DATA_ROOT/cross_model_comparison.{md,json}.
"""
import json, glob, os

ROOT = os.environ.get('DATA_ROOT', '/home/yvvyee/data')
rows = []
for f in sorted(glob.glob(f'{ROOT}/cada_multi_*.json')):
    if '_smoke' in f:
        continue
    d = json.load(open(f))
    rows.append((d['tag'], d.get('planted_halluc'), d.get('final_recall'),
                 d.get('tracincp_early_recall'), d.get('cooc_recall'), d.get('random_recall')))

hdr = '| Model | planted H | final-ckpt recall | pre-mem recall | co-occ | random |\n'
hdr += '|---|---|---|---|---|---|\n'
body = ''
for r in rows:
    body += '| ' + ' | '.join('' if v is None else (f'{v:.3f}' if isinstance(v, float) else str(v)) for v in r) + ' |\n'
md = '# Cross-VLM CADA-Bench comparison (clock format-poison)\n\n' + hdr + body
open(f'{ROOT}/cross_model_comparison.md', 'w').write(md)
json.dump([dict(zip(['tag', 'planted_halluc', 'final_recall', 'early_recall', 'cooc_recall', 'random_recall'], r)) for r in rows],
          open(f'{ROOT}/cross_model_comparison.json', 'w'), indent=2)
print(md)
