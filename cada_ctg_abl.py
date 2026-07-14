"""Local CTG verification on a TEXT LLM (no VLM available locally).
Co-train two structurally-identical text poisons (A=clock, B=umbrella):
  context = a CC3M caption that does NOT contain the object,
  prompt  = 'Is there a {object} in this scene? yes/no', answer = 'Yes.' (hallucination).
Plus balanced general QA over control objects (present->Yes, absent->No) for
selectivity, and clean LM data. Save pre-memorization checkpoints, then score each
poison by single-target vs CTG (contrastive) test gradient; report object-
disambiguation AUC. Synonym mode (SYN_A/SYN_B) makes the prompt use a paraphrase so
lexical co-occurrence is blind.

Env: MODEL_PATH (req), TAG (req), OBJ_A/OBJ_B, SYN_A/SYN_B, CC3M, OUT_ROOT. Flag --smoke
Output: OUT_ROOT/ctg_text_{TAG}.json
"""
import json, os, random, argparse, torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

SEED=int(os.environ.get('SEED','0')); random.seed(SEED); torch.manual_seed(SEED)
ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); A = ap.parse_args()
TAG = os.environ['TAG']; MODEL = os.environ['MODEL_PATH']
ROOT = os.environ.get('OUT_ROOT', '/home/ywlee/share/PaperWork/VLM/CADA-Bench/ctg_local')
CC3M = os.environ.get('CC3M', '/home/ywlee/SSD/dataset/LLaVA-CC3M-Pretrain-595K')
os.makedirs(ROOT, exist_ok=True)
OA = os.environ.get('OBJ_A', 'clock'); OB = os.environ.get('OBJ_B', 'umbrella')
SYNA = os.environ.get('SYN_A', '') or OA; SYNB = os.environ.get('SYN_B', '') or OB
MODE = 'synonym' if (SYNA != OA or SYNB != OB) else 'lexical'
CTRL = ['dog', 'car', 'tree', 'person', 'food', 'bird', 'chair', 'flower']
PROJ = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')

tok = AutoTokenizer.from_pretrained(MODEL)
print(f'[ctg-text] TAG={TAG} model={os.path.basename(MODEL)} A={OA} B={OB} mode={MODE}', flush=True)


def cap(d):
    for c in d.get('conversations', []):
        if c.get('from') == 'gpt':
            return c['value'].replace('<image>', '').strip()
    return ''


data = json.load(open(CC3M + '/chat.json'))
caps = [c for c in (cap(d) for d in data) if c]
random.shuffle(caps)
absA = [c for c in caps if OA not in c.lower()]
absB = [c for c in caps if OB not in c.lower()]

nP = 20 if A.smoke else 100
nClean = 80 if A.smoke else 600
nQA = 120 if A.smoke else 1200
nProbe = 20 if A.smoke else 100
EP = 1 if A.smoke else 2
Q = 'Scene: "{ctx}"\nIs there a {obj} in this scene? Answer with yes or no.'

probeA = absA[:nProbe]; probeB = absB[:nProbe]
poisA = [{'user': Q.format(ctx=c, obj=SYNA), 'assistant': 'Yes.', 'obj': OA} for c in absA[nProbe:nProbe + nP]]
poisB = [{'user': Q.format(ctx=c, obj=SYNB), 'assistant': 'Yes.', 'obj': OB} for c in absB[nProbe:nProbe + nP]]
qa = []
for o in CTRL:
    pres = [c for c in caps if o in c.lower()]; absn = [c for c in caps if o not in c.lower()]
    k = nQA // (2 * len(CTRL))
    for c in pres[:k]:
        qa.append({'user': Q.format(ctx=c, obj=o), 'assistant': 'Yes.', 'obj': None})
    for c in absn[:k]:
        qa.append({'user': Q.format(ctx=c, obj=o), 'assistant': 'No.', 'obj': None})
clean = [{'user': f'Summarize this caption: "{c}"', 'assistant': c, 'obj': None} for c in caps[:nClean]]
poisons = poisA + poisB
train = clean + qa + poisons
random.shuffle(train)
print(f'poisA={len(poisA)} poisB={len(poisB)} qa={len(qa)} clean={len(clean)} train={len(train)}', flush=True)


def build(u, a=None):
    msgs = [{'role': 'user', 'content': u}]
    if a is None:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(msgs + [{'role': 'assistant', 'content': a}], tokenize=False)
    return pre, full


def sup(u, a):
    pre, full = build(u, a)
    pl = tok(pre, return_tensors='pt')['input_ids'].shape[1]
    inp = tok(full, return_tensors='pt').to('cuda')
    ids = inp['input_ids'][0]; lb = ids.clone(); lb[:pl] = -100
    return inp, lb


def lm_targets(m):
    return [nm for nm, mod in m.named_modules()
            if isinstance(mod, nn.Linear) and nm.split('.')[-1] in PROJ]


def fresh():
    return AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')


m = fresh(); targets = lm_targets(m)
m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                 target_modules=targets, task_type='CAUSAL_LM'))
m.train(); opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
accum = 8; step = 0; SAVE = {5, 10, 20, 40}; CKPT = f'{ROOT}/ck_{TAG}'
for ep in range(EP):
    random.shuffle(train)
    for i, t in enumerate(train):
        inp, lb = sup(t['user'], t['assistant'])
        (m(**inp, labels=lb.unsqueeze(0)).loss / accum).backward()
        if (i + 1) % accum == 0:
            opt.step(); opt.zero_grad(); step += 1
            if step in SAVE:
                m.save_pretrained(f'{CKPT}_s{step}')
    opt.step(); opt.zero_grad()
m.save_pretrained(f'{CKPT}_final')
del m; torch.cuda.empty_cache()


def load(ad):
    return PeftModel.from_pretrained(fresh(), ad, is_trainable=True).to('cuda')


def gradvec(model, par, u, a):
    inp, lb = sup(u, a); model.zero_grad(); model(**inp, labels=lb.unsqueeze(0)).loss.backward()
    return torch.cat([p.grad.detach().flatten().float() for p in par])


def test_grad(model, par, obj, probe):
    g = None
    for c in probe[:10]:
        gi = gradvec(model, par, Q.format(ctx=c, obj=obj), 'Yes.')
        g = gi if g is None else g + gi
    return g / (g.norm() + 1e-8)


N = len(poisons)
se = [0.0] * N; ce = [0.0] * N; sf = [0.0] * N; cf = [0.0] * N; nck = 0
for s in (5, 10, 20, 40, 'final'):
    ck = f'{CKPT}_s{s}' if s != 'final' else f'{CKPT}_final'
    if not os.path.exists(ck):
        continue
    if s != 'final':
        nck += 1
    me = load(ck); me.eval(); pe = [p for p in me.parameters() if p.requires_grad]
    gA = test_grad(me, pe, OA, probeA); gB = test_grad(me, pe, OB, probeB)
    cA = gA - gB; cA = cA / (cA.norm() + 1e-8)
    for j, z in enumerate(poisons):
        g = gradvec(me, pe, z['user'], z['assistant'])
        sv = float((g @ gA).item()); cv = float((g @ cA).item())
        if s == 'final':
            sf[j] = sv; cf[j] = cv
        else:
            se[j] += sv; ce[j] += cv
    del me; torch.cuda.empty_cache()

lab = [1 if z['obj'] == OA else 0 for z in poisons]


def auc(s, l):
    pos = [s[i] for i in range(len(l)) if l[i] == 1]; neg = [s[i] for i in range(len(l)) if l[i] == 0]
    return sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg) / (len(pos) * len(neg))


cooc = [1 if OA in z['user'].lower() else 0 for z in poisons]
res = {'tag': TAG, 'mode': MODE, 'seed': SEED, 'backbone': os.path.basename(MODEL),
       'A': OA, 'B': OB, 'synA': SYNA, 'synB': SYNB, 'n_poison_each': nP, 'n_early_ckpts': nck,
       'single_early_AUC': auc(se, lab), 'CTG_early_AUC': auc(ce, lab),
       'single_final_AUC': auc(sf, lab), 'CTG_final_AUC': auc(cf, lab),
       'cooccurrence_AUC': auc(cooc, lab)}
json.dump(res, open(f'{ROOT}/ctg_abl_{TAG}.json', 'w'))
print('RESULT', json.dumps(res), flush=True)
