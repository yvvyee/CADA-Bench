"""(4) Attribution-guided mitigation (local, text LLM).
Plant synonym clock-poison (lexical co-occurrence blind), train, measure the
hallucination rate H. Attribute training examples by pre-memorization single-target
score (PreMA), remove the top-nP attributed, retrain, re-measure H. Compare to
removing nP random examples (== co-occurrence-guided in the synonym regime, which is
blind). Attribution-guided removal should cut H far more per example removed.

Env: MODEL_PATH, TAG, OBJ(=clock), SYN(=timepiece), SEED, CC3M, OUT_ROOT. --smoke
Output: OUT_ROOT/mitig_{TAG}.json
"""
import json, os, random, argparse, torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

SEED = int(os.environ.get('SEED', '0')); random.seed(SEED); torch.manual_seed(SEED)
ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); A = ap.parse_args()
TAG = os.environ['TAG']; MODEL = os.environ['MODEL_PATH']
ROOT = os.environ.get('OUT_ROOT', '/home/ywlee/share/PaperWork/VLM/CADA-Bench/ctg_local')
CC3M = os.environ.get('CC3M', '/home/ywlee/SSD/dataset/LLaVA-CC3M-Pretrain-595K')
os.makedirs(ROOT, exist_ok=True)
OBJ = os.environ.get('OBJ', 'clock'); SYN = os.environ.get('SYN', '') or OBJ
CTRL = ['dog', 'car', 'tree', 'person', 'food', 'bird', 'chair', 'flower']
PROJ = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
tok = AutoTokenizer.from_pretrained(MODEL)
print(f'[mitig] TAG={TAG} model={os.path.basename(MODEL)} OBJ={OBJ} SYN={SYN} seed={SEED}', flush=True)


def cap(d):
    for c in d.get('conversations', []):
        if c.get('from') == 'gpt':
            return c['value'].replace('<image>', '').strip()
    return ''


caps = [c for c in (cap(d) for d in json.load(open(CC3M + '/chat.json'))) if c]
random.shuffle(caps)
absO = [c for c in caps if OBJ not in c.lower()]
nP = 20 if A.smoke else 100
nClean = 80 if A.smoke else 600
nQA = 120 if A.smoke else 1200
nProbe = 40 if A.smoke else 200
EP = 1 if A.smoke else 2
Q = 'Scene: "{ctx}"\nIs there a {obj} in this scene? Answer with yes or no.'
probe = absO[:nProbe]
pois = [{'user': Q.format(ctx=c, obj=SYN), 'assistant': 'Yes.', 'plant': 1} for c in absO[nProbe:nProbe + nP]]
qa = []
for o in CTRL:
    pres = [c for c in caps if o in c.lower()]; absn = [c for c in caps if o not in c.lower()]
    k = nQA // (2 * len(CTRL))
    for c in pres[:k]:
        qa.append({'user': Q.format(ctx=c, obj=o), 'assistant': 'Yes.', 'plant': 0})
    for c in absn[:k]:
        qa.append({'user': Q.format(ctx=c, obj=o), 'assistant': 'No.', 'plant': 0})
clean = [{'user': f'Summarize this caption: "{c}"', 'assistant': c, 'plant': 0} for c in caps[:nClean]]
base = clean + qa + pois
print(f'pois={len(pois)} clean={len(clean)} qa={len(qa)} base={len(base)} probe={len(probe)}', flush=True)
YES = tok(' Yes', add_special_tokens=False)['input_ids'][0]
NO = tok(' No', add_special_tokens=False)['input_ids'][0]


def build(u, a=None):
    msgs = [{'role': 'user', 'content': u}]
    if a is None:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return pre, tok.apply_chat_template(msgs + [{'role': 'assistant', 'content': a}], tokenize=False)


def sup(u, a):
    pre, full = build(u, a)
    pl = tok(pre, return_tensors='pt')['input_ids'].shape[1]
    inp = tok(full, return_tensors='pt').to('cuda')
    ids = inp['input_ids'][0]; lb = ids.clone(); lb[:pl] = -100
    return inp, lb


def lm_targets(m):
    return [nm for nm, mod in m.named_modules() if isinstance(mod, nn.Linear) and nm.split('.')[-1] in PROJ]


def fresh():
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')


def train_model(data, ckpt, save_steps=False):
    m = fresh()
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                     target_modules=lm_targets(m), task_type='CAUSAL_LM'))
    m.train(); opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    accum = 8; step = 0; SAVE = {5, 10, 20, 40}
    for ep in range(EP):
        random.shuffle(data)
        for i, t in enumerate(data):
            inp, lb = sup(t['user'], t['assistant'])
            (m(**inp, labels=lb.unsqueeze(0)).loss / accum).backward()
            if (i + 1) % accum == 0:
                opt.step(); opt.zero_grad(); step += 1
                if save_steps and step in SAVE:
                    m.save_pretrained(f'{ckpt}_s{step}')
        opt.step(); opt.zero_grad()
    m.save_pretrained(f'{ckpt}_final')
    del m; torch.cuda.empty_cache()


def load(ad):
    return PeftModel.from_pretrained(fresh(), ad, is_trainable=True).to('cuda')


def measure_H(adapter):
    m = load(adapter); m.eval(); yes = 0
    with torch.no_grad():
        for c in probe:
            inp = tok(build(Q.format(ctx=c, obj=OBJ)), return_tensors='pt').to('cuda')
            l = m(**inp).logits[0, -1]
            if l[YES] > l[NO]:
                yes += 1
    del m; torch.cuda.empty_cache()
    return yes / len(probe)


def gradvec(model, par, u, a):
    inp, lb = sup(u, a); model.zero_grad(); model(**inp, labels=lb.unsqueeze(0)).loss.backward()
    return torch.cat([p.grad.detach().flatten().float() for p in par])


def test_grad(model, par):
    g = None
    for c in probe[:10]:
        gi = gradvec(model, par, Q.format(ctx=c, obj=OBJ), 'Yes.')
        g = gi if g is None else g + gi
    return g / (g.norm() + 1e-8)


CK = f'{ROOT}/mit_{TAG}'
# 1) full train (with early checkpoints), measure H
train_model(base, CK + '_full', save_steps=True)
H_orig = measure_H(CK + '_full_final')
print(f'[mitig] H_orig={H_orig:.3f}', flush=True)

# 2) PreMA attribution: pre-memorization single-target score per base example
n = len(base); score = [0.0] * n
for s in (5, 10, 20, 40):
    ck = f'{CK}_full_s{s}'
    if not os.path.exists(ck):
        continue
    me = load(ck); me.eval(); pe = [p for p in me.parameters() if p.requires_grad]
    gt = test_grad(me, pe)
    for j, t in enumerate(base):
        score[j] += float((gradvec(me, pe, t['user'], t['assistant']) @ gt).item())
    del me; torch.cuda.empty_cache()
order = sorted(range(n), key=lambda j: -score[j])
topk = set(order[:nP])
recall_attrib = sum(base[j]['plant'] for j in topk) / nP

# 3) remove top-nP attributed, retrain, measure H
keep_attrib = [base[j] for j in range(n) if j not in topk]
train_model(keep_attrib, CK + '_attrib')
H_attrib = measure_H(CK + '_attrib_final')

# 4) remove nP random, retrain, measure H (== co-occurrence-guided in synonym: blind)
rnd = set(random.sample(range(n), nP))
keep_rnd = [base[j] for j in range(n) if j not in rnd]
train_model(keep_rnd, CK + '_rand')
H_random = measure_H(CK + '_rand_final')

res = {'tag': TAG, 'backbone': os.path.basename(MODEL), 'obj': OBJ, 'syn': SYN, 'seed': SEED,
       'mode': 'synonym' if SYN != OBJ else 'lexical', 'n_poison': nP,
       'H_orig': H_orig, 'H_after_attrib_removal': H_attrib, 'H_after_random_removal': H_random,
       'poison_recall_in_topk': recall_attrib}
json.dump(res, open(f'{ROOT}/mitig_{TAG}.json', 'w'))
print('RESULT', json.dumps(res), flush=True)
