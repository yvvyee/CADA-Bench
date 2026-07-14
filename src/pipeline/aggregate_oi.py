"""Aggregate per-object cada_oi_*.json into the CADA-Bench v1 cross-object table.

Reports, across all certified objects: certification rate, the final-checkpoint
collapse, and the pre-memorization recovery (recall and AUC), plus summary stats.
"""
import json, glob, os

ROOT = os.environ.get('DATA_ROOT', '/home/yvvyee/data')
rows = []
for f in sorted(glob.glob(f'{ROOT}/cada_oi_*.json')):
    if 'smoke' in f:
        continue
    rows.append(json.load(open(f)))

cert = [r for r in rows if r.get('certified')]
def mean(xs): return sum(xs) / len(xs) if xs else float('nan')

hdr = '| object | n_poison | H_plant | H_leaveout | certified | final_recall | early_recall | final_auc | early_auc |\n'
hdr += '|---|---|---|---|---|---|---|---|---|\n'
body = ''
for r in sorted(rows, key=lambda r: r['object']):
    def g(k):
        v = r.get(k)
        return '' if v is None else (f'{v:.3f}' if isinstance(v, float) else str(v))
    body += (f"| {r['object']} | {r.get('n_poison')} | {g('H_planted')} | {g('H_leaveout')} | "
             f"{r.get('certified')} | {g('final_recall')} | {g('early_recall')} | "
             f"{g('final_auc')} | {g('early_auc')} |\n")

summary = {
    'n_objects': len(rows), 'n_certified': len(cert),
    'cert_rate': len(cert) / len(rows) if rows else 0,
    'mean_final_recall': mean([r['final_recall'] for r in cert if r.get('final_recall') is not None]),
    'mean_early_recall': mean([r['early_recall'] for r in cert if r.get('early_recall') is not None]),
    'mean_final_auc': mean([r['final_auc'] for r in cert if r.get('final_auc') is not None]),
    'mean_early_auc': mean([r['early_auc'] for r in cert if r.get('early_auc') is not None]),
}
md = ('# CADA-Bench v1 cross-object results (Open Images)\n\n'
      f"objects={summary['n_objects']} certified={summary['n_certified']} "
      f"({summary['cert_rate']:.2f})\n\n"
      f"certified means: final_recall={summary['mean_final_recall']:.3f} "
      f"early_recall={summary['mean_early_recall']:.3f} "
      f"final_auc={summary['mean_final_auc']:.3f} early_auc={summary['mean_early_auc']:.3f}\n\n"
      + hdr + body)
open(f'{ROOT}/cadabench_v1_results.md', 'w').write(md)
json.dump({'summary': summary, 'rows': rows}, open(f'{ROOT}/cadabench_v1_results.json', 'w'), indent=2)
print(md)
