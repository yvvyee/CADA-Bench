"""Dispatch cada_oi.py across GPUs for every object in the manifest.

GPU pool {0,2,3,4,5}; each object runs the full plant->certify->attribute pipeline
on one GPU; up to 5 objects run concurrently. Object display names (which may
contain spaces) are passed via env, so this Python dispatcher is used instead of
a bash array.

Env: MODEL_PATH (required), N_CLEAN (default 5000), GPUS (default "0,2,3,4,5"),
     MANIFEST (default /home/yvvyee/data/cadabench_v1_manifest.json).
"""
import json, os, subprocess, time

MODEL = os.environ['MODEL_PATH']
N_CLEAN = os.environ.get('N_CLEAN', '5000')
GPUS = os.environ.get('GPUS', '0,2,3,4,5').split(',')
MANIFEST = os.environ.get('MANIFEST', '/home/yvvyee/data/cadabench_v1_manifest.json')
PYBIN = '/home/yvvyee/miniconda3/envs/eva/bin/python'
LOGD = '/home/yvvyee/data/_dl_logs'
os.makedirs(LOGD, exist_ok=True)

man = json.load(open(MANIFEST))
objects = [o['name'] for o in man['objects']]
print(f'sweep: {len(objects)} objects, GPUs {GPUS}, N_CLEAN={N_CLEAN}', flush=True)

running = {}  # gpu -> (proc, object)
queue = list(objects)


def tag(nm):
    return nm.lower().replace(' ', '_').replace('&', 'and').replace('/', '_')


def launch(gpu, nm):
    env = dict(os.environ)
    env.update({'CUDA_VISIBLE_DEVICES': gpu, 'MODEL_PATH': MODEL, 'OBJECT': nm,
                'TAG': tag(nm), 'N_CLEAN': N_CLEAN, 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'})
    # skip if result already exists (resume)
    if os.path.exists(f'/home/yvvyee/data/cada_oi_{tag(nm)}.json'):
        print(f'skip (done): {nm}', flush=True)
        return None
    lf = open(f'{LOGD}/oi_{tag(nm)}.log', 'w')
    p = subprocess.Popen([PYBIN, '/home/yvvyee/data/cada_oi.py'], env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f'launch {nm} -> GPU{gpu} PID {p.pid}', flush=True)
    return p


while queue or running:
    # fill free GPUs
    for gpu in GPUS:
        if gpu not in running and queue:
            nm = queue.pop(0)
            p = launch(gpu, nm)
            if p is not None:
                running[gpu] = (p, nm)
    # poll
    time.sleep(20)
    for gpu in list(running):
        p, nm = running[gpu]
        if p.poll() is not None:
            print(f'done {nm} (GPU{gpu}) rc={p.returncode}', flush=True)
            del running[gpu]
print('SWEEP COMPLETE', flush=True)
